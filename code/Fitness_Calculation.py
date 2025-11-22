from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from numpy.random import randint 
from math import sqrt
import numpy as np

def sol_chk(ss,n_ft):
    ss = [randint(0,len(ss)) for i in range(n_ft)]
    return np.unique(ss)

def wrapper_KNN(feat,label):
    k=3;
    X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=12345)
      
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    train_preds = knn_model.predict(X_train)
    mse = mean_squared_error(y_train, train_preds)
    rmse = sqrt(mse)
    return rmse

def Fitness_function(selfea,label,X,ws):
    ws = [0.99, 0.01];
    error    = wrapper_KNN(selfea,label);
    # % Number of selected features
    num_feat = len(sum(X == 1));
    # % Total number of features
    max_feat = len(X); 
    alpha    = ws[0]; 
    beta     = ws[1];
    # % Cost function 
    cost     = alpha * error + beta * (num_feat / max_feat);
    return cost
