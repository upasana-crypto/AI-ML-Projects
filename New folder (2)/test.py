import numpy as np


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit([[1, 2], [np.nan, 3], [7, 6]])

X = [[1, 2], [np.nan, 3], [7, 6]]
print(imp.transform(X))
