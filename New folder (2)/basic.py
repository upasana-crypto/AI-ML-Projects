from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

X_train=pd.read_csv('train_pid1.csv')
y_train=pd.read_csv('train_label.csv')
X_test=pd.read_csv('test_pid0.csv')

logreg=LogisticRegression()
X_train=X_train[:1][1:15]
#y_train=np.zeros(1,15)
y_train=y_train[:1][1:15]

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)