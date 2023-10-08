#!/usr/bin/env python
# coding: utf-8

# # DAT200 CA5 2022
# 
# Kaggle username: aarondeleyos

# ### Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from scipy import stats
from sklearn.utils.fixes import loguniform
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ### Reading data

# In[2]:


training_data = pd.read_csv(r'data\train.csv')
training_data = training_data.drop('Unnamed: 0', axis=1)


# ### Data exploration and visualisation

# In[3]:


# Check dataset for any missing values
training_data.info()


# In[4]:


training_data.describe()


# In[5]:


for column in training_data.columns:
    print(f'{column}: {(training_data[column] == 0).sum()}')


# For the feature 'Vitamin C', the 25th and 50th percentile consist of only zeroes since most of its values are equal to zero. Also the minimum value for the 'Scoville score' is 221 while the mean of the featuer is in the range of $ 10^6 $. Perhaps a typo?

# In[6]:


# Pair plot of the data
plt.figure(figsize=(16, 9))
sns.pairplot(data=training_data)
plt.show()


# In[7]:


# Pair plot of the data log-transformed
plt.figure(figsize=(16, 9))
sns.pairplot(data=np.log(training_data + 1))
plt.show()


# In[8]:


# Correlation matrix
corr_matrix = training_data.corr()
plt.figure(figsize=(16, 9))
sns.heatmap(data=corr_matrix, annot=True)
plt.show()


# ### Data cleaning

# In[9]:


# Remove outliers
cleaned_data = training_data.copy()
cleaned_data = cleaned_data[(np.abs(stats.zscore(cleaned_data)) < 3).all(axis=1)]
cleaned_data.index = range(cleaned_data.shape[0])


# In[10]:


# Remove the data point where
idx = int(np.where(cleaned_data['Scoville score'] == np.min(cleaned_data['Scoville score']))[0])
cleaned_data = cleaned_data.drop(index=idx, axis=1)
cleaned_data.index = range(cleaned_data.shape[0])


# ### Data exploration after cleaning

# In[11]:


# Pair plot after removing outliers
plt.figure(figsize=(16, 9))
sns.pairplot(data=cleaned_data)
plt.show()


# ### Data preprocessing

# In[12]:


X = cleaned_data.iloc[:, :-1]
y = cleaned_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3,
                                                    random_state=1)


# ### Modelling

# In[13]:


def eval_metrics(y_true, y_pred, output_str):
    print(5*'-', output_str, 5*'-')
    print(f'{output_str} score (MAE): {mean_absolute_error(y_true, y_pred):.4f}' )
    print(f'{output_str} score (R^2): {r2_score(y_true, y_pred):.4f}')
    print(f'{output_str} score (RMSE): {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}')
    print()


# #### Data pipeline with regression model

# In[14]:


# Random forest regression model
rfr_pipe = make_pipeline(StandardScaler(),
                         RandomForestRegressor(criterion='absolute_error',
                                               random_state=163,
                                               n_jobs=-1))


# In[15]:


# Nested cross-validation for the model

inner_segments = 10
outer_segments = 10

rfr_grid = [{'randomforestregressor__max_depth': [3, 5, 10],
             'randomforestregressor__max_features': ['sqrt', 'log2', None]}]

rfr_gs = GridSearchCV(estimator=rfr_pipe,
                      param_grid=rfr_grid,
                      scoring='neg_mean_absolute_error',
                      cv=inner_segments,
                      n_jobs=-1)

scores_nested_rfr = cross_val_score(rfr_gs, X_train, y_train, 
                                    scoring='neg_mean_absolute_error', 
                                    cv=outer_segments)

print(f'CV accuracy: {np.mean(scores_nested_rfr):.3f} +/- {np.std(scores_nested_rfr):.3f}')


# In[16]:


# Fit the model with the 'best' parameters to the training data
rfr_model = rfr_gs.fit(X_train, y_train)
print(rfr_model.best_params_)

# Use the model to predict on the training and test set
y_train_pred = rfr_model.predict(X_train)
y_test_pred = rfr_model.predict(X_test)

# Evaluate the MAE of the predicted values compared to the 'ground truth'
eval_metrics(y_train, y_train_pred, 'Training')
eval_metrics(y_test, y_test_pred, 'Testing')


# #### Data pipeline with classification model

# ##### Binning train target values
# 
# Can be performed with ex. pandas.qcut or pandas.cut
# 
# ```python
# n_bins = 10
# y_train_binned = pd.cut(y_train, n_bins, labels=False) # or
# y_train_binned = pd.qcut(y_train, n_bins, labels=False) 
# ```

# In[17]:


xgb_clf = xgb.XGBClassifier(objective='multi:softprob', random_state=2, 
                            verbosity=0, eval_metric='mae')


# In[18]:


def bin_converter(y_binned, bins):
    """
    Converts the predicted class labels (0, 1, 2, etc.) into the relating value
    in the bins.
    """
    for i, bin_value in enumerate(y_binned):
        y_binned[i] = bins[bin_value]


# In[19]:


for n_bins in [15, 10, 5]:
    y_train_binned, bins = pd.cut(y_train, n_bins, labels=False, retbins=True)
    inner_segments = 10

    # Parameter grid for the model
    xgb_grid = [{'xgbclassifier__learning_rate': [x/10 for x in range(1, 4)],
                'xgbclassifier__gamma': [10**x for x in range(-4, 3)],
                'xgbclassifier__subsample': [0.5, 0.7],
                'xgbclassifier__max_depth':[1, 3, 5, 7],
                }]

    # Randomized grid search
    xgb_gs = RandomizedSearchCV(estimator=xgb_clf,
                                param_distributions=xgb_grid, 
                                scoring='neg_mean_absolute_error', 
                                cv=inner_segments, n_iter=10,
                                n_jobs=-1)

    scores_nested_xgb = cross_val_score(xgb_gs, X_train, y_train_binned, 
                                        scoring='neg_mean_absolute_error', 
                                        cv=outer_segments)

    xgb_classifier = xgb_gs.fit(X_train, y_train_binned)

    # print(xgb_classifier.best_params_)

    # Need to convert y_train_binned into a numpy
    # array to easier convert the bins into actual values later in the for loop
    y_train_binned = np.array(y_train_binned)

    # Predictions on training and test set
    y_train_pred = xgb_classifier.predict(X_train)
    y_test_pred = xgb_classifier.predict(X_test)

    # We have to bin the ground truth for the test set
    # with the help of the bins created in the training set.
    y_test_binned = np.digitize(y_test, bins=bins, right=True)

    # Convert all the predicted class labels into bin values
    for binned_y in [y_train_pred, y_test_pred, y_test_binned, y_train_binned]:
        binned_y = bin_converter(binned_y, bins)

    # MAE on the training and test set
    eval_metrics(y_train_binned, y_train_pred, f'Training, num bins: {n_bins}')
    eval_metrics(y_test_binned, y_test_pred, f'Testing,  num bins: {n_bins}')


# The classification model's performance worsens as the number of bins decrease.

# #### Other models used for Kaggle submission

# #### Regression model #2

# In[20]:


X_log = np.log(X + 1) # Log-transform on the features
y_sqrt = np.sqrt(y) # Square-root transform on the target

X_train_log, X_test_log, y_train, y_test = train_test_split(X_log, y_sqrt, test_size=0.3,
                                                            random_state=12)


# In[21]:


xgbreg_pipe = make_pipeline(xgb.XGBRegressor(eval_metric='mae', random_state=2))


# In[22]:


inner_segments = 12
outer_segments = 12

# Parameter grid for the model
xgb_grid = [{'xgbregressor__learning_rate':loguniform(0.01, 0.4).rvs(20, random_state=1),
             'xgbregressor__gamma': loguniform(0.01, 10).rvs(20, random_state=1),
             'xgbregressor__subsample': np.linspace(0.1, 1, 10),
             'xgbregressor__max_depth': [1, 3, 5, 7],
             'xgbregressor__alpha': loguniform(0.01, 0.3).rvs(20, random_state=1),
             'xgbregressor__lambda': loguniform(0.01, 0.3).rvs(20, random_state=2)}]

# Randomized grid search
xgb_gs = RandomizedSearchCV(xgbreg_pipe, xgb_grid, 
                            scoring='neg_mean_absolute_error', 
                            cv=inner_segments, n_iter=30, n_jobs=-1)

scores_xgb = cross_val_score(xgb_gs, X_train_log, y_train, 
                             scoring='neg_mean_absolute_error',
                             cv=outer_segments, n_jobs=-1)

# print(f'CV MAE: {np.mean(scores_xgb):.3f} +/- {np.std(scores_xgb):.3f}')


# In[23]:


# Fit the model on the training data
xgb_regressor = xgb_gs.fit(X_train_log, y_train)

# Get the parameters from the 'best' model
best_params = xgb_regressor.best_params_
print(best_params)

# Make predictions on the training and test set
y_train_pred = xgb_regressor.predict(X_train_log)
y_test_pred = xgb_regressor.predict(X_test_log)

# Evaluate the MAE of the predicted values compared to the 'ground truth'.
# We have to square our predictions and ground truth as we square-root transformed the target
eval_metrics(y_train**2, y_train_pred**2, f'Training')
eval_metrics(y_test**2, y_test_pred**2, f'Testing')


# In[24]:


# Create a model based on the best parameters
xgb_model = xgb.XGBRegressor(subsample=best_params['xgbregressor__subsample'], 
                             max_depth=best_params['xgbregressor__max_depth'],
                             learning_rate=best_params['xgbregressor__learning_rate'],
                             reg_lambda=best_params['xgbregressor__lambda'],
                             gamma=best_params['xgbregressor__gamma'],
                             alpha=best_params['xgbregressor__alpha'],
                             random_state=2)
xgb_model = xgb_model.fit(X_train_log, y_train)

# Plot the feature importance to evaluate whether or not some features are redundant
fig, ax = plt.subplots(figsize=(10, 5))
xgb.plot_importance(xgb_model, ax=ax)
plt.show()


# In[25]:


# Get the y_ticklabels in ascending order, as shown in the plot above
y_ticklabels = ax.get_yticklabels()
features_sorted = []

for yticklabel in y_ticklabels:
    features_sorted.append(yticklabel.get_text())

# We then select which of the features we're going to remove, going bottom up 
removed_features = features_sorted[:6] 


# In[26]:


# We then make a copy of our log-transformed data and remove the unwanted features
X_log_refined = X_log.copy()
X_log_refined = X_log.drop(removed_features, axis=1)

X_train_log, X_test_log, y_train, y_test = train_test_split(X_log_refined, y_sqrt, test_size=0.3,
                                                            random_state=12)


# In[27]:


xgb_model = xgb.XGBRegressor(subsample=best_params['xgbregressor__subsample'], 
                             max_depth=best_params['xgbregressor__max_depth'],
                             learning_rate=best_params['xgbregressor__learning_rate'],
                             reg_lambda=best_params['xgbregressor__lambda'],
                             gamma=best_params['xgbregressor__gamma'],
                             alpha=best_params['xgbregressor__alpha'])

# By the help of the model with the 'best' parameters we now fit our new data onto it
# and make predictions
xgb_model = xgb_model.fit(X_train_log, y_train)
y_train_pred = xgb_model.predict(X_train_log)
y_test_pred = xgb_model.predict(X_test_log)

# Then we evaluate the MAE the same as before.
eval_metrics(y_train**2, y_train_pred**2, f'Training')
eval_metrics(y_test**2, y_test_pred**2, f'Testing')


# ### Final Evaluation

# ### Kaggle submission

# In[30]:


test = pd.read_csv(r'data\test.csv')
test = test.drop('Unnamed: 0', axis=1)


# In[31]:


# We log-transform the test_data as we did the training data
test_log = np.log(test + 1)

# We remove the features which where not used to train our final model 
test_log = test_log.drop(removed_features, axis=1)

# We make predictions and square them
predictions = xgb_model.predict(test_log)**2

my_submission = pd.DataFrame({'Id': list(range(len(predictions))),
                              'Scoville score': predictions})
my_submission.to_csv(r'submissions/XGBRegressor18_log_sqrt_removedFewFeatures.csv', index=False)


# In[ ]:




