import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#from sklearn.datasets import load_breast_cancer
#cancer=pd.read_csv('E:\Programming\GoCode_Local\GoCode\Breast_Cancer_data')
#cancer.head()

from sklearn.datasets import load_breast_cancer
cancer= load_breast_cancer()
print(cancer.keys())
x=cancer.data
y=cancer.feature_names
df=pd.DataFrame(x, columns=cancer.feature_names)
print(df.head())

