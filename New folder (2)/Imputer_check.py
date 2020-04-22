import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

X={'Index':[1,2],'Price':[10,20],'Age':[24,25]}
label={'Label':[0,1]}
label=pd.Series(label)
test={'Index':[3],'Price':[10],'Age':[25]}
test=pd.DataFrame.from_dict(test)
print(type(label))
new_index=list(X.keys())
x_dict=pd.DataFrame.from_dict(X)
# print(X)
# X=pd.DataFrame(X)
# print(X)
# k=0
# X=np.array(X)
# # print(X)
# imp=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# imp.fit(X)
# X[k:k+2,:]=imp.transform(X[k:k+2,:])

print(x_dict)
print(test)

y=pd.DataFrame()
index_list=list(x_dict['Index'].unique())
y['Index']=index_list
for column in x_dict.iloc[:,1:].columns:
    y[column]=(x_dict.groupby('Index')[column].apply(list).tolist())
    

# y['Index']=index_list
# print(y.index)
print(y)


# combined_training=pd.merge(y,label,left_on='Index',right_on='Index',how='right')
# print(combined_training)
# p=pd.DataFrame()
# p=y
# t=combined_training['Label']
# print(t)
# pipe=OneVsRestClassifier(LinearSVC(random_state=0))
# pipe.fit(p['Price'],t).predict(test)

# OneVsRestClassifier(LinearSVC).fit(y,label).predict(test)
# pipe.fit(y,label)

# y_enc = MultiLabelBinarizer().fit(y)
# clf = OneVsRestClassifier(SVC())
# clf.fit(y_enc,label)
# predictions = clf.predict_proba(test)
# print(predictions)

# my_metrics = metrics.classification_report( test_y, predictions)
# print my_metrics
 #Default hyperparameters

y_pred=SVC().fit(x_dict,y).predict(test)

