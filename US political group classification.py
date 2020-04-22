import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

plt.style.use('ggplot')  # R language's ggplot library used to plot graphics with data for 

iris=datasets.load_iris()
type(iris)

print(iris.keys())
x=iris.data
y=iris.target

#print(x)
#print(y)

df=pd.DataFrame(x, columns=iris.feature_names) # without columns feature row will have 0 1 2 3 etc. with columns it gives the actual names of the features
print(df.head())
print(df.shape)