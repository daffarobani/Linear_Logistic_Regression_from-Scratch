# -*- coding: utf-8 -*-
"""
Tutorial Data Sciences and Statistics
Multiple Logistic Regression
Author: M. Daffa Robani
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats import weightstats as stests
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})
from mpl_toolkits.mplot3d import Axes3D


#Import the Data from .csv file
data = pd.read_csv('ex2data1.txt',header=None)
#data = pd.read_csv('diabetes.csv')
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

pos = np.where(y==1) #To find which row(s) that have y=1
neg = np.where(y==0) #To find which row(s) that have y=0

#Visualizing the data with scatter plot
#You can d this if your data contains 1 or 2 variable(s)
#Call the datascatterplot_log function with input: X,y,n_var,pos,neg
def datascatterplot_log(X,y):
    pos = np.where(y==1) #To find which row(s) that have y=1
    neg = np.where(y==0) #To find which row(s) that have y=0
    n_var = X.shape[1]
    if n_var==1:
        fig = plt.figure(figsize=[8,6])
        plt.scatter(X[pos],y[pos])
        plt.scatter(X[neg],y[neg])
        plt.show()
    elif n_var==2:
        fig = plt.figure(figsize=[8,6])
        plt.scatter(X[pos,0],X[pos,1],label='y=1')
        plt.scatter(X[neg,0],X[neg,1],label='y=0')
        plt.xlabel('x1',fontsize=14)
        plt.ylabel('x2',fontsize=14)
        plt.title('Scatter Plot of The Samples',fontsize=16)
        plt.legend()
        plt.show()
    else:
        print("Your data has more than two variables")
datascatterplot_log(X,y)

def sigmoid(scores):
    #Function to calculate sigmoid function based on scores at each input
    return 1/(1+np.exp(-scores))

def log_likelihood_cal(X,y,beta):
    F = np.ones((X.shape[0],X.shape[1]+1))
    F[:,1:] = X
    scores = F@beta
    log_likelihood = np.sum(y*scores-np.log(1+np.exp(scores)))
    return log_likelihood
        
def log_likelihood_der_cal(X,y,beta):
    F = np.ones((X.shape[0],X.shape[1]+1))
    F[:,1:] = X
    scores = F@beta
    dl = np.sum(F*(y-sigmoid(scores)).reshape(-1,1),axis=0)
    
#    dl2 = (F.T@F)*(sigmoid(scores))*(1-sigmoid(scores))
    temp = np.ones((F.shape[1],F.shape[1],F.shape[0]))
    for i in range(X.shape[0]):
        temp[:,:,i] = np.multiply(F[i,:].reshape(-1,1)@F[i,:].reshape(-1,1).T,-1*sigmoid(scores[i])*(1-sigmoid(scores[i])))
    dl2 = np.sum(temp,axis=2)
    return dl,dl2

def logistic_regression_train(X,y):
    #Initial values
    beta = np.zeros((X.shape[1]+1))
    
    likelihood_old = 0
    likelihood_now = log_likelihood_cal(X,y,beta)
    [dl,dl2] = log_likelihood_der_cal(X,y,beta)
    it = 0
    tol = 1e-6
    while np.abs(likelihood_now-likelihood_old) > tol:
        it = it+1
        beta = beta-np.linalg.inv(dl2)@dl
        likelihood_old = np.copy(likelihood_now)
        likelihood_now = log_likelihood_cal(X,y,beta)
        [dl,dl2] = log_likelihood_der_cal(X,y,beta)
        print("Iteration: {}, Likelihood:{}".format(it,likelihood_now))
        
    se = np.sqrt(np.diag(np.linalg.inv(-dl2)))
    return beta,likelihood_now, se

[beta,likelihood,beta_se] = logistic_regression_train(X,y)
def summarymat_log(beta, beta_se):
    n_var = beta.shape[0]-1
    summary = np.zeros((n_var+1,5))
    summary[:,0] = beta
    summary[:,1] = beta_se #Standard deviation of each foefficient
    summary[:,2] = np.divide(summary[:,0],summary[:,1]) #z-statistic
    summary[:,3] = stats.norm.sf(np.abs(summary[:,2]))*2
    #Hypothesis test with 95% confidence interval
    for i in range(n_var+1):
        if summary[i,3]<= 0.05:
            summary[i,4] = 1
        else:
            summary[i,4] = 0
    return summary

summary = summarymat_log(beta, beta_se)

def pseudo_r2_comp(y,likelihood):
    pos = np.where(y==1) #To find which row(s) that have y=1
    neg = np.where(y==0) #To find which row(s) that have y=0
#    Calculate the log odds
    odds = pos[0].shape[0]/neg[0].shape[0]
#    log_odds = np.log(pos[0].shape[0]/neg[0].shape[0])
    p_odds = odds/(1+odds)
    #Calculate the pseudo TSS
    tss = np.sum(y*np.log(p_odds)+(1-y)*np.log(1-p_odds))
    #Calculate the pseudo R2
    R2 = (tss-likelihood)/tss
    return R2

R2 = pseudo_r2_comp(y,likelihood)

def modelplot(beta,X,y):
    if X.shape[1]==1:
        X_plot = np.linspace(30,100,51)
        F = np.ones((51,2))
        F[:,1] = X_plot
        y_plot = sigmoid(F@beta)
        fig = plt.figure(figsize=[8,6])
        plt.scatter(X,y,label='Samples')
        plt.plot(X_plot,y_plot,color='black',label='Logistic Regression Model')
        plt.title('Logistic Regression Model on The Samples',fontsize=16)
        plt.xlabel('x',fontsize=14)
        plt.ylabel('y',fontsize=14)
        plt.legend()
        plt.show
    elif X.shape[1] == 2:
        m = -1*beta[1]/beta[2]
        c = -1*beta[0]/beta[2]-np.log(1)/beta[2]
        
        x_bound = np.linspace(30,100,51)
        y_bound = m*x_bound+c
        
        pos = np.where(y==1) #To find which row(s) that have y=1
        neg = np.where(y==0) #To find which row(s) that have y=0
        fig = plt.figure(figsize=[8,6])
        plt.scatter(X[pos,0],X[pos,1],label='y=1')
        plt.scatter(X[neg,0],X[neg,1],label='y=0')
        plt.plot(x_bound,y_bound,color='black',label='Boundary Line')
        plt.title('Predicted Boundary Line on The Samples',fontsize=16)
        plt.xlabel('x1',fontsize=14)
        plt.ylabel('x2',fontsize=14)
        plt.legend()
        plt.show()
    else:
        print("Your data has more than two variables")
    
modelplot(beta,X,y)




    





    
    
