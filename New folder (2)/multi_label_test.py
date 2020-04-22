import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier

# train_features=pd.read_csv('train_pid1.csv')
# X_train_feature=pd.DataFrame(train_features)
# print(X_train_feature.head())
# train_labels=pd.read_csv('train_label.csv')
# X_train_labels=pd.DataFrame(train_labels)
# print(X_train_labels.head())

X={'Price':[10,20,30,40,50,60],'Age':[24,25,26,27,28,29]}
X=pd.DataFrame(X)
label={'Label1':[0,1,0,0,1,1],'Label2':[1,0,1,1,0,0]}
label=pd.DataFrame(label)
# print(label)
label1=np.array(label)
# label1=np.ravel(label1)
test={'Price':[20,10,60,40],'Age':[25,24,29,27]}
test=pd.DataFrame.from_dict(test)
# x=X['Price']
# y=label['Label']
X1=np.array(X)
test=np.array(test)
print(type(X1))
print(type(label1))
print(X1)
print(label1)



y_pred=MultiOutputClassifier(SVC()).fit(X1,label1)
# p=np.array(y_pred.decision_function(test))

# p = np.array(clf.decision_function(X)) # decision is a voting function
# prob=1/(1+np.exp(-p))
# prob = np.exp(p)/np.sum(np.exp(p),axis=1) # softmax after the voting
classes = y_pred.predict(test)
# prob=1/(1+np.exp(-classes))
# h=y_pred.predict(test)
# print(y_pred)
# print(type(y_pred))
# print(abs(prob))
# print(theta.dtype)
print(classes)
# print(type(p))
# i=0
# for i in range(0,len(p)):
#     if p[i]<0:
#         p[i]=1+p[i]

# print(p)
