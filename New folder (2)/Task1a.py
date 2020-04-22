import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import Ridge

import math

from filefuns import *

data = csv2numpy('train.csv',pre_ones=0,wanted_labels=[])

kf = sk.model_selection.KFold(n_splits=10,shuffle=False)

lambdas = [0.01, 0.1, 1, 10, 100]
averages = []

for lbd in lambdas:
    l2regressor = Ridge(alpha=lbd, normalize=False, fit_intercept=False)

    RMSE_sum = 0

    for train_index, test_index in kf.split(data):

        trainingdata = data[train_index]
        train_labels = trainingdata[:,0]
        train_features = trainingdata[:,1:]

        valdata = data[test_index]
        val_labels = valdata[:,0]
        val_features = valdata[:,1:]

        
        l2regressor = l2regressor.fit(train_features,train_labels)

        predicted_labels = l2regressor.predict(val_features)

        RMSE_sum += math.sqrt(sk.metrics.mean_squared_error(val_labels, predicted_labels))

    RMSE = (1/kf.get_n_splits()) * RMSE_sum
    averages.append(RMSE)

print(averages)
numpy2csv(np.array(averages),'results.csv')

