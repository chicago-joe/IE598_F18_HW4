
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 00:53:49 2018

Author: Joseph Loss
"""

# Loading the Housing Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Read dataframe and print header
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book/master/code/'
                 'datasets/housing/housing.data', 
                 header=None, 
                 sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
            'NOX', 'RM', 'AGE', 'DIS', 'RAD',
            'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

print(df.head())
print(df.describe())


# EDA - Using scatterplots to visualize pairwise correlations of different features in dataset
cols = df.columns
sns.pairplot(df[cols], height=2.5)
plt.tight_layout()
plt.show()


# EDA - Using Seaborn's heatmap to plot a correlation matrix array of features
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
            cbar=True,
            annot=False,
            square=True,
            fmt='.2f',
            annot_kws={'size': 15},
            yticklabels=cols,
            xticklabels=cols)
plt.suptitle("Feature Heat Map", fontsize=20)
plt.show()


# Split data into test and training sets using all variables in the dataset
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
slr = LinearRegression()
slr.fit(X_train, y_train)
slr_y_train_pred = slr.predict(X_train)
slr_y_test_pred = slr.predict(X_test)


# Plot residual vs. predicted values to diagnose the regression model
plt.scatter(slr_y_train_pred, slr_y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(slr_y_test_pred, slr_y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.suptitle("Linear Regression Diagnostic", fontsize=20)
plt.show()

# Print Slope and y-Intercept
print('Slope: %.3f' % slr.coef_[0])
print('y-Intercept: %.3f' % slr.intercept_)

# Compute MSE of training and test predictions
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, slr_y_train_pred),
        mean_squared_error(y_test, slr_y_test_pred)))
# Compute R2 of training and test datasets
print('R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, slr_y_train_pred),
       r2_score(y_test, slr_y_test_pred)))

# =============================================================================
# Ridge Regression Model
# Defining the cv_score plot
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)
    
    std_error = cv_scores_std / np.sqrt(10)
    
    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    plt.suptitle("CV Scores vs. Alpha Analysis")
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
    
# Fit Ridge Regression Model over a range of different alphas and plot cv-R2 
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

ridge = Ridge(normalize=True)
ridge.fit(X_train, y_train)
ridge_y_train_pred = ridge.predict(X_train)
ridge_y_test_pred = ridge.predict(X_test)

ridge.score(X_test, y_test)

for alpha in alpha_space:
    ridge.alpha = alpha
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    ridge_scores.append(np.mean(ridge_cv_scores))
    ridge_scores_std.append(np.std(ridge_cv_scores))
display_plot(ridge_scores, ridge_scores_std)


# Plot residual vs. predicted values to diagnose the regression model
plt.scatter(ridge_y_train_pred, ridge_y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(ridge_y_test_pred, ridge_y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.suptitle('Ridge Regression Diagnostic')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

# Print Slope and y-Intercept
print('Slope: %.3f' % ridge.coef_[0])
print('y-Intercept: %.3f' % ridge.intercept_)

# Compute MSE of training and test predictions
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, ridge_y_train_pred),
        mean_squared_error(y_test, ridge_y_test_pred)))
# Compute R2 of training and test datasets
print('R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, ridge_y_train_pred),
       r2_score(y_test, ridge_y_test_pred)))
print('The optimal Ridge Alpha is:', alpha_space[np.argmax(ridge_scores)])

# =============================================================================
# LASSO Regression
# Fit LASSO Regression Model over a range of different alphas and plot cv-R2 
lasso_alpha_space = np.logspace(-4, 0, 50)
lasso_scores = []
lasso_scores_std = []

lasso = Lasso()
for alpha in lasso_alpha_space:
    lasso.alpha = alpha
    lasso_cv_scores = cross_val_score(lasso, X, y, cv=10)
    lasso_scores.append(np.mean(lasso_cv_scores))
    lasso_scores_std.append(np.std(lasso_cv_scores))
display_plot(lasso_scores, lasso_scores_std)

lasso.fit(X_train, y_train).coef_
lasso_y_train_pred = lasso.predict(X_train)
lasso_y_test_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)

# Plot residual vs. predicted values to diagnose the regression model
plt.scatter(lasso_y_train_pred, lasso_y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(lasso_y_test_pred, lasso_y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.suptitle('LASSO Regression Diagnostic')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

# Print Slope and y-Intercept
print('Slope: %.3f' % lasso.coef_[0])
print('y-Intercept: %.3f' % lasso.intercept_)

# Compute MSE of training and test predictions
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, lasso_y_train_pred),
        mean_squared_error(y_test, lasso_y_test_pred)))
# Compute R2 of training and test datasets
print('R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, lasso_y_train_pred),
       r2_score(y_test, lasso_y_test_pred)))
print('The optimal Lasso Alpha is:', lasso_alpha_space[np.argmax(lasso_scores)])


# EXTRA: LASSO Regression for feature selection (housing prices) ============
names = df.drop('MEDV', axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_

_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
_ = plt.suptitle('BONUS: LASSO Regression - Feature Selection') 
plt.show()


# ElasticNet Regression Model
# Compute train and test errors
def elanet_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(l1_space, cv_scores)
    
    std_error = cv_scores_std / np.sqrt(10)
    
    ax.fill_between(l1_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('L1_Space')
    ax.set_title('ElasticNet CV Analysis',fontsize=20)
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([l1_space[0], l1_space[-1]])
    ax.set_xscale('log')
    plt.show()

elanet = ElasticNet()
elanet.fit(X_train, y_train)
elanet_y_train_pred = elanet.predict(X_train)
elanet_y_test_pred = elanet.predict(X_test)

elanet_scores = []
elanet_scores_std = []

l1_space = np.logspace(-4,0,50)
for l1_ratio in l1_space:
    elanet.l1_ratio = l1_ratio
    elanet_cv_scores = cross_val_score(elanet, X, y, cv=10)
    elanet_scores.append(np.mean(elanet_cv_scores))
    elanet_scores_std.append(np.std(elanet_cv_scores))

elanet_plot(elanet_scores, elanet_scores_std)

plt.figure()
plt.scatter(elanet_y_train_pred, elanet_y_train_pred - y_train,c='steelblue', marker='o',
            edgecolor='white',label='Training data')
plt.scatter(elanet_y_test_pred, elanet_y_test_pred - y_test,c='limegreen', marker='s',
            edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.suptitle('ElasticNet Regression Diagnostic')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

# Print Slope and y-Intercept
print('Slope: %.3f' % elanet.coef_[0])
print('y-Intercept: %.3f' % elanet.intercept_)

# Compute MSE of training and test predictions
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, elanet_y_train_pred),
        mean_squared_error(y_test, elanet_y_test_pred)))
# Compute R2 of training and test datasets
print('R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, elanet_y_train_pred),
       r2_score(y_test, elanet_y_test_pred)))
print('The optimal l1_ratio is:', l1_space[np.argmax(elanet_scores)])

# =============================================================================
print()
print("=======================================================================")
print("My name is Joseph Loss")
print("My NetID is: loss2")
print("I hereby certify that I have read the University policy on Academic Integrity"
      " and that I am not in violation.")
# =============================================================================