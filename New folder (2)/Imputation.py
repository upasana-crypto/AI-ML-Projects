from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

def Imput(Data):
    X_train=pd.DataFrame(Data)
    no_of_patients=X_train['pid'].unique()
    total_entries=len(no_of_patients)
    X=X_train.to_numpy()
    s=0
    for j in range(0,total_entries):
        s=s+12
        for q in range(0,len(X_train.columns)):
            c=0
            p=0
            for p in range(0,12):
                X[p+s-12,1]=p+1 #
                if np.isnan(X[p+s-12,q]) == True:
                    c=c+1
                    # print(c)
            if c==12:
                 for p in range(0,12):
                     X[p+s-12,q]=0.0
  
    k=0
    for i in range(0,total_entries):
        k=12+k
        imp=SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(X[k-12:k,:])
        X[k-12:k,:]=imp.transform(X[k-12:k,:])

    X = pd.DataFrame(X)
    X.columns=X_train.columns
    return X