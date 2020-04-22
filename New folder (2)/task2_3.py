from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import Ridge

from Imputation import Imput

train_feature=pd.read_csv('train_pid1.csv')
train_labels=pd.read_csv('training_label.csv')
test_feature=pd.read_csv('test_pid0.csv')

y_train=pd.DataFrame(train_labels)
y_train=y_train.iloc[:,1:]

result=Imput(train_feature)
test=Imput(test_feature)
print('imputation_finished')

# Here is our manipulation funciton
def manipulate(Data1):
    train_array_mean=pd.DataFrame()
    train_array_median=pd.DataFrame()
    train_array_max_min=pd.DataFrame()

    total_entries=len(Data1['Age'])//12
    # print(total_entries)
    s=0
    
    for k in range(0,total_entries):
        s=s+12
        test_mean=[]
        test_median=[]
        test_max_min=[]
       
        for q in range(0,len(Data1.columns)):
            #we created three separate series here to further evaluate the features
            test_mean.append(Data1.iloc[s-12:s,q].mean())
            test_median.append(Data1.iloc[s-12:s,q].median())
            test_max_min.append((Data1.iloc[s-12:s,q].max())-(Data1.iloc[s-12:s,q].min()))
        
        test_mean=pd.Series(test_mean)
        test_median=pd.Series(test_median[3:])
        test_max_min=pd.Series(test_max_min[3:])

        train_array_mean=train_array_mean.append(test_mean,ignore_index=True)
        train_array_median=train_array_median.append(test_median,ignore_index=True)
        train_array_max_min=train_array_max_min.append(test_max_min,ignore_index=True)

    train_or_test_array=pd.concat([train_array_mean,train_array_median,train_array_max_min],axis=1,ignore_index=True)
    # train_or_test_array=train_or_test_array
    print('manupulation is going on')
    return train_or_test_array

train_array=manipulate(result) # pass the dataframe with columns removed
# test_array=manipulate(test)
# train_array.to_csv('train_array.csv',index=False)

train_array=train_array.to_numpy()
# test_array=test_array.to_numpy()
# y_train_labels=y_train.to_numpy()
# print(y_train)


scaler = StandardScaler()
train_norm=scaler.fit_transform(train_array)
# print(type(train_norm))


pca = PCA(n_components=15)
prin_comp=pca.fit_transform(train_norm)
# print(pca.singular_values_.shape)
# train_final=pca.singular_values_

principal_comps = pd.DataFrame(data = prin_comp)
X=principal_comps

# parameters = {'kernel':('linear', 'rbf')'C':[1, 10]'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
# searcher = GridSearchCV(clf,scoring='roc_auc', cv=10,return_train_score=True)

kf=KFold(n_splits=10, random_state=None, shuffle=True)

lambdas = [0.01, 0.1, 1, 10, 100]
averages = []

for lbd in lambdas:
    l2regressor = Ridge(alpha=lbd, normalize=False, fit_intercept=False)

    RMSE_sum = 0
    for train_index, val_index in kf.split(X):
  
        X_train_final, X_val_final = X.iloc[train_index], X.iloc[val_index]
    
        y_train_final, y_val_final = y_train.iloc[train_index], y_train.iloc[val_index]
    
        l2regressor = l2regressor.fit(X_train_final,y_train_final)

        predicted_labels = l2regressor.predict(X_val_final)

        RMSE_sum += math.sqrt(sk.metrics.mean_squared_error(y_val_final, predicted_labels))

        RMSE = (1/kf.get_n_splits()) * RMSE_sum
        averages.append(RMSE)

print(averages)



