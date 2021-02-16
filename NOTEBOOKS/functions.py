# Import necessary libraries/packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import itertools
from sklearn import metrics

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV, cross_val_score)

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
                              
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
RandomForestRegressor)

from sklearn.metrics import (classification_report, confusion_matrix, 
                             plot_confusion_matrix, precision_score, 
                             accuracy_score, recall_score, f1_score, roc_curve, 
                             auc)

from scipy.special import logit

from functions import *

plt.style.use('seaborn')

import shap
shap.initjs()

from alibi.explainers import KernelShap
from scipy.special import logit

from sklearn.feature_extraction.text import TfidfVectorizer



# Define functions that will be repeatedly used 

def model_performance(model, X_train, X_test, y_train, y_test, pred):
    ''' This function will display the model's accuracy scores, classification
        report, and plot the model's confusion matrix'''
    
    # Plot confusion matrix
    plot_confusion_matrix(model, X_test, y_test, normalize='true', cmap='Blues',
                          display_labels=['Not Popular (0)', 'Popular (1)'])
    
    # Create a classification report to display evaluation metrics
    print("Classification Report \n", classification_report(y_test, pred, 
                                                            target_names=['Not Popular',
                                                            'Popular']))
    
    print("\n")
    print("----------------------------------------------------------")
    
    # Training, Testing accuracy scores
    print("Training Accuracy Score: {:.4}%".format(model.score(X_train, y_train) * 100))
    print("Testing Accuracy Score: {:.4}%".format(model.score(X_test, y_test) * 100))
    print("Accuracy: {:.4}%".format(metrics.accuracy_score(y_test, pred) * 100))
    
    print("\n")
    print("----------------------------------------------------------")
    
def plot_feature_importances(model, X_train, X):
    ''' This function will plot the feature importances of the model '''
    
    # Create barplot of feature importances
    sns.set(style="darkgrid")
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), X_train.columns.values) 
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    
    # Returns a chart of features and their importance values
    feature_imp = pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
    return feature_imp


def find_coeffs(model, X_train, X):
    ''' This function will determine coefficient values and format them
        in a dataframe for easy interpretation'''
    
    coeffs = pd.Series(model.coef_[0], index=X_train.columns)
    return pd.DataFrame(coeffs, 
             X.columns, 
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)

def roc_auc(model, X_train, X_test, y_train, y_test):
    ''' This function will plot the ROC curve and display AUC for the model '''
    
    # Calculate the probability scores of each point in the training and test set
    y_train_score = model.decision_function(X_train)
    y_test_score = model.decision_function(X_test)
    
    # Calculate the fpr, tpr, and thresholds for the training and test set
    train_fpr, train_tpr, thresholds = roc_curve(y_train, y_train_score)
    test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_score)
    
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

    # ROC curve for training set
    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(train_fpr, train_tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve for Training Set')
    plt.legend(loc='lower right')
    plt.show();
    
    # ROC curve for training set
    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(test_fpr, test_tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve for Test Set')
    plt.legend(loc='lower right')
    plt.show();
    
    # ROC curve for both training and test set
    plt.figure(figsize=(10, 8))
    lw = 2

    plt.plot(train_fpr, train_tpr, color='blue',
             lw=lw, label='Train ROC curve')
    plt.plot(test_fpr, test_tpr, color='darkorange',
             lw=lw, label='Test ROC curve')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show();
    
    print('Train AUC: {}'.format(auc(train_fpr, train_tpr)))
    print('Test AUC: {}'.format(auc(test_fpr, test_tpr)))
    
    
    
    
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

    
def roc_dt_rf(y_test, pred, label):    
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)

    # ROC curve for training set
    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr, tpr, label=label,
              color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show();
    
    print('AUC: {}'.format(auc(fpr, tpr)))
    
def build_sm_ols(df, features, target, add_constant=False):
    ''' This function builds OLS model and prints out summary '''
    X = pd.DataFrame(df[features])
    if add_constant:
        X = sm.add_constant(X)
    y =  df[target]
    ols = sm.OLS(y, X).fit()
    return ols

def plot_residuals(ols):
    '''This function plots model residual distribution '''
    sns.set(style="darkgrid")
    residuals = ols.resid
    plt.figure(figsize=(8,5))
    plt.title('Residuals Distribution')
    sns.distplot(residuals)
    plt.show();
    
    print('\n')
    
    plt.figure()
    x_axis = np.linspace(0, 1, len(residuals))
    plt.scatter(x_axis, residuals)
    plt.title('Residuals and Baseline')
    plt.show();
    
def qqplot(model, title):
    ''' This function creates a qq plot  '''
    fig, ax = plt.subplots(figsize=(8, 5))
    fig = sm.graphics.qqplot(model.resid, line='45', fit=True, ax=ax)
    plt.title(title)
    plt.show();
    
def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
 
def plot_shap(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train)
    shap.summary_plot(shap_values, X_train, plot_type='bar')
    
    
def plot_shap_tree(model, X_train, X, nsamples=100):
    X = shap.utils.sample(X_train, nsamples=nsamples)
    explainer = shap.TreeExplainer(model, X)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
    shap.summary_plot(shap_values, X, plot_type='bar')