# -*- coding: utf-8 -*-
"""
Tutorial Data Sciences and Statistics
Multiple Linear Regression
Author: M. Daffa Robani
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})
from mpl_toolkits.mplot3d import Axes3D



#Import the Data from .csv file
data = pd.read_csv('Advertising_2D.csv',index_col='Unnamed: 0')
data_array = data.to_numpy() #Convert to numpy array
n_entries = data_array.shape[0] #Check the number of entries (number of samples)
n_var = data_array.shape[1]-1 #Check the number of variables


X = data.iloc[:,0:n_var].to_numpy() #Take X (features) from the data
y = data.iloc[:,-1].to_numpy() #Take y (responses) from the data


##Make correlation matrix of each variable
def correlmat(data,n_var):
    correl = np.zeros((n_var+1,n_var+1))
    for i in range(n_var+1):
        for j in range(n_var+1):
            correl[i,j],_ = stats.pearsonr(data[:,i],data[:,j]) #Calculate (pearson) correlation between each variables
    return correl

#Call the correlmat function with input: data_array and n_var
correl_matrix = correlmat(data_array,n_var)

#Visualizing the data with scatter plot
#You can do this if the data only contains 1 or 2 variable(s)
def datascatterplot(X,y,n_var):
    if n_var == 1:
        fig = plt.figure(figsize=[8,6])
        plt.scatter(X,y,color='tab:orange')
        plt.title('Scatter Plot of the Samples',fontsize=16)
        plt.xlabel('x',fontsize=14)
        plt.ylabel('y',fontsize=14)
        plt.grid()
        plt.show()
    elif n_var == 2:
        fig = plt.figure(figsize=[8,6])
        ax = plt.axes(projection='3d')
        ax.scatter3D(X[:,0],X[:,1],y,color='tab:orange')
        ax.set_xlabel('x1',fontsize=14)
        ax.xaxis.labelpad = 10
        ax.set_ylabel('x2',fontsize=14)
        ax.yaxis.labelpad = 10
        ax.set_zlabel('y',fontsize=14)
        plt.title('Scatter Plot of the Samples',fontsize=16)
        plt.show()
    else:
        print("Your data has more than two variables")

#Call the datascatterplot with input: x, y, n_var
datascatterplot(X,y,n_var)

#Creating linear regression model
#Create F matrix 
F = np.ones((n_entries,n_var+1))
F[:,1:] = X

def LinearRegression(F,y):
    n_entries = F.shape[0]
    n_var = F.shape[1]
    #Regression Coefficient(s)
    beta = np.linalg.inv(np.transpose(F)@F)@np.transpose(F)@y #Minimizng RSS procedure
    #beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(F),F)),np.transpose(F)),y) #Minimizng RSS procedure
    #Variances of the regression coefficient
    y_hat = F@beta
    #y_hat = np.matmul(F,beta) #Calculate prediction at training locations
    res_sqr = np.power((y-y_hat),2) #Calculate squared residuals
    sig_hat_sqr = np.sum(res_sqr)/(n_entries-n_var) #Estimate variance of the prediction model
    beta_var = (np.linalg.inv(np.transpose(F)@F)*sig_hat_sqr).diagonal()
    #beta_var = (np.linalg.inv(np.matmul(np.transpose(F),F))*sig_hat_sqr).diagonal() #Calculate variance of each regression coefficient
    return beta,beta_var

#Call LinearRegression with input: F,y to obtain beta(Regression coefficient) and beta_var (variance of each coefficient)
beta, beta_var = LinearRegression(F,y)

##For validation (compare with scikit)
from sklearn.linear_model import LinearRegression as LinearRegression_1
reg = LinearRegression_1().fit(X,y)

beta_scikit = np.ones(n_var+1)
beta_scikit[1:] = reg.coef_
beta_scikit[0] = reg.intercept_

#For validation (compare with statsmodels)
import statsmodels.api as sm
ols = sm.OLS(y, F)
ols_result = ols.fit()
summary_ols = ols_result.summary


#Create summary matrix for the coefficient, std.error, z statistic, p value, and hypothesis test

def summarymat(beta, beta_var, n_entries):
    n_var = beta.shape[0]-1
    beta_std = np.power(beta_var,0.5)
    summary = np.zeros((n_var+1,5))
    summary[:,0] = beta #Beta values (Regression coefficient)
    summary[:,1] = beta_std #Standard deviation of each foefficient
    summary[:,2] = np.divide(beta,beta_std) #t-statistic
    df = n_entries-2 #Degree of freedom (number of entries-2)
    summary[:,3] = (np.ones((summary.shape[0],1))-stats.t.cdf(summary[:,2],df).reshape(-1,1)).reshape(-1,) #Calculate p-value from t statistic
    #Hypothesis test with 95% confidence interval
    for i in range(n_var+1):
        if summary[i,3]<= 0.05:
            summary[i,4] = 1
        else:
            summary[i,4] = 0
    return summary

#Call summarymat function with input: beta, beta_var, n_entries to obtain the summary matrix
summary = summarymat(beta,beta_var,n_entries)

#Assessing the accuracy of the model
def r2_comp(F,y,beta):
    TSS = np.sum(np.power((y-np.mean(y)),2)) #Calculate TSS(Total sum of squares) from the samples
    #Calculate RSS (Residual sum of squares)
    y_hat = F@beta
    #y_hat = np.matmul(F,beta) #Prediction at training locations
    RSS = np.sum(np.power((y-y_hat),2))
    r2 = (TSS-RSS)/TSS #Calculate r2
    return r2

#Call r2_comp with input: F, y, beta to obtain r2 score of the model
r2 = r2_comp(F,y,beta)

#Visualize our linear regression model
#Range of each variable to visualize
x1range = np.array([0,300])
x2range = np.array([0,50])
def modelplot(beta,X,y,x1range,x2range):
    if X.shape[1] == 2: #Two Dimensional Case
        #Create prediction location
        x1, x2= np.meshgrid(np.linspace(x1range[0],x1range[1],21),np.linspace(x2range[0],x2range[1],21)) 
        x1_vec = x1.reshape(-1,)
        x2_vec = x2.reshape(-1,)
        F_temp = np.ones((21*21,3))
        F_temp[:,1] = x1_vec
        F_temp[:,2] = x2_vec
        #Calculate predictions at determined location
        y_hat_vec = F_temp@beta
        #y_hat_vec = np.matmul(F_temp,beta)
        y_hat_mat = y_hat_vec.reshape(21,21)
        
        fig = plt.figure(figsize=[8,6])
        ax = fig.gca(projection='3d')
        ax.scatter3D(X[:,0],X[:,1],y,color='tab:orange',label='Samples')
        surf = ax.plot_surface(x1,x2,y_hat_mat,color='tab:blue',label='Linear Regression Model')
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d
        ax.set_xlabel('x1',fontsize=14)
        ax.xaxis.labelpad = 10
        ax.set_ylabel('x2',fontsize=14)
        ax.yaxis.labelpad = 10
        ax.set_zlabel('y',fontsize=14)
        plt.title('Linear Regression Model on The Samples',fontsize=16)
        ax.legend(loc=2,bbox_to_anchor=(0.1, 0.9))
        plt.show()
    
    elif X.shape[1] == 1: #One Dimensional Case
        #Create prediction location
        x1 = np.linspace(x1range[0],x1range[1],21)
        F_temp = np.ones((21,2))
        F_temp[:,1] = x1
        #Calculate predictions at determined location
        y_hat = np.matmul(F_temp,beta)
        
        fig = plt.figure(figsize=[8,6])
        plt.scatter(X,y,color='tab:orange', label='Samples')
        plt.plot(x1,y_hat,color='tab:blue',linewidth=4, label='Linear Regression Model')
        plt.title('Linear Regression Model on The Samples', fontsize=16)
        plt.xlabel('x',fontsize=14)
        plt.ylabel('y',fontsize=14)
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("Your data has more than two variables")
        
#Call modelplot function to visualize the Linear Regression Model
modelplot(beta,X,y,x1range,x2range)
    
    
    
    
    
    
    






