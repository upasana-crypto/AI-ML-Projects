import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


train_feature=pd.read_csv('train_pid1.csv')
train_labels=pd.read_csv('train_label.csv')
test_features=pd.read_csv('test_pid0.csv')
#X_train=train_features.reset_index()
#y_train=train_labels.reset_index()
#X_test=test_features.reset_index()
X_train=pd.DataFrame(train_feature)
y_train=pd.DataFrame(train_labels)
X_test=pd.DataFrame(test_features)
#print(df.isnull().sum())
#print(X_train.head(18))
trainingData=X_train.iloc[:,:].values
dataset=X_train.iloc[:,:].values
# Setup the pipeline steps: steps :Try changing the strategy types to check if result improves
#steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='most')),('SVM', SVC())]
imp=SimpleImputer(missing_values=np.nan, strategy='mean')
# Create the pipeline: pipeline
#pipeline = Pipeline(steps)

imp=imp.fit(trainingData[:,:])

X_new=imp.transform()

X_train=df.groupby(by ='pid', axis=0).mean()

print(X_train)
svm=SVC()
svm.fit(X_train,y_train)

# Predict the labels of the test set
y_pred = svm.predict(X_test)

# Compute metrics
print(classification_report(y_test,y_pred))
