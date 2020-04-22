import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from Imputation import Imput

train_feature=pd.read_csv('train_pid1.csv')
train_labels=pd.read_csv('training_label.csv')
test_feature=pd.read_csv('test_pid0.csv')

y_train=pd.DataFrame(train_labels)
y_train=y_train.iloc[:,1]

# result=Imput(train_feature).iloc[:,1:]
# test=Imput(test_feature).iloc[:,1:]

result=Imput(train_feature)
test=Imput(test_feature)
print('imputation_finished')
# Here is our manipulation funciton
def manipulate(Data1):
    train_array_mean=pd.DataFrame()
    train_array_median=pd.DataFrame()
    train_array_max_min=pd.DataFrame()

    total_entries=len(Data1['Age'])//12
    print(total_entries)
    s=0
    
    for k in range(0,total_entries):
        s=s+12
        test_mean=[]
        test_median=[]
        test_max_min=[]
        # print(test)
        for q in range(0,len(Data1.columns)):
            #we created three separate series here to further evaluate the features
            test_mean.append(Data1.iloc[s-12:s,q].mean())
            test_median.append(Data1.iloc[s-12:s,q].median())
            test_max_min.append((Data1.iloc[s-12:s,q].max())-(Data1.iloc[s-12:s,q].min()))
        
        test_mean=pd.Series(test_mean)
        test_median=pd.Series(test_median)
        test_max_min=pd.Series(test_max_min)

        train_array_mean=train_array_mean.append(test_mean,ignore_index=True)
        train_array_median=train_array_median.append(test_median,ignore_index=True)
        train_array_max_min=train_array_max_min.append(test_max_min,ignore_index=True)

    train_or_test_array=pd.concat([train_array_mean, train_array_median,train_array_max_min],axis=1)
    print('manupulation is going on')
    return train_or_test_array

train_array=manipulate(result) # pass the dataframe with columns removed
# test_array=manipulate(test)
# train_array.to_csv('train_array.csv',index=False)
train_array=train_array.to_numpy()
# test_array=test_array.to_numpy()
y_train_labels=y_train.to_numpy()


# print(train_array)
# print(y_train_labels)
# print(train_array.shape)
# print(test_array.shape)
# # print(train_array.shape)
# y_train=y_train.iloc[:,1].to_numpy(dtype=float)
# y_train=np.array([1,1,0,1,0,0])
# y_true_label=np.array([1,1,0])


# train_norm = StandardScaler(train_array)
# print(type(train_norm))
# test_norm = StandardScaler(test_array)
# X_train, X_test, y_train_new, y_test = train_test_split(train_array, y_train, test_size=0.4, random_state=0)

clf=SVC(C=1)
# clf=LinearSVC(penalty='l2',loss='hinge',dual=True, random_state=32)
# print(X_array.shape)
# scores = cross_val_score(clf, train_array, y_train_labels, cv=5)
# print(scores)
# clf.fit(X_train,y_train_new)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(clf, parameters,scoring='roc_auc', cv=10,return_train_score=True)
searcher.fit(train_array,y_train)

# Report the best parameters
print("Best CV params", searcher.best_params_)

y_test_label=searcher.predict(train_array)
print(y_test_label)
print(searcher.best_score_)

# print(pd.DataFrame(searcher.cv_results_['mean_train_r2','mean_test_r2']))
# accuracy=accuracy_score(y_test,y_test_label)
# print(accuracy)
# Z=pd.merge(X_train_feature,y_train,left_on='pid',right_on='pid',how='right')
# print(Z)



