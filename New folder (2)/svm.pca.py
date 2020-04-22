from sklearn.decomposition import PCA
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
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from Imputation import Imput

train_feature=pd.read_csv('train_pid1.csv')
train_labels=pd.read_csv('training_label.csv')
test_feature=pd.read_csv('test_pid0.csv')

y_train=pd.DataFrame(train_labels)
y_train=y_train.iloc[:,1:]
# print(y_train.shape)
# print(y_train)

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
    # print(total_entries)
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

# print(train_array.head(2))
# train_norm = StandardScaler(train_array)
# print(type(train_norm))
# test_norm = StandardScaler(test_array)

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
# print(X)
# print(X.shape)
# print(principal_comps.head())


# X_train, X_test, y_train_new, y_test = train_test_split(train_array, y_train, test_size=0.4, random_state=0)

clf=MultiOutputClassifier(SVC())
# clf=SVC(C=1)
# clf=LinearSVC(penalty='l2',loss='hinge',dual=True, random_state=32)
# print(X_array.shape)
# scores = cross_val_score(clf, train_array, y_train_labels, cv=5)
# print(scores)
# clf.fit(X_train,y_train_new)

# parameters = {'kernel':('linear', 'rbf')'C':[1, 10]'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
# searcher = GridSearchCV(clf,scoring='roc_auc', cv=10,return_train_score=True)
# Z=pd.merge(X,y_train,left_on='pid',right_on='pid',how='right')
# print(Z)
mean_score=0
kf=KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, val_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    print('hi')
    X_train_final, X_val_final = X.iloc[train_index], X.iloc[val_index]
    # print(X_train_final,'\n',X_test_final)
    y_train_final, y_val_final = y_train.iloc[train_index], y_train.iloc[val_index]
    clf.fit(X_train_final,y_train_final)
    predicted_labels=clf.predict(X_val_final)
    # score=clf.score(predicted_labels,y_val_final)
    # mean_score=mean_score+score
    # mean_score=roc_auc_score(predicted_labels,y_val_final,average='micro')

# print(mean_score)

fpr = dict()
tpr = dict()
roc_auc = dict()
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(2):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 2

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# searcher = cross_val_score(clf,principal_comps,y_train,cv=10)
# searcher.fit(principal_comps,y_train)

# Report the best parameters
# print("Best CV params", searcher.best_params_)

# y_test_label=searcher.predict(principal_comps)
# print(y_test_label)
# print(searcher.best_score_)
# print(searcher)

# print(pd.DataFrame(searcher.cv_results_['mean_train_r2','mean_test_r2']))
# accuracy=accuracy_score(y_test,y_test_label)
# print(accuracy)
# 



