from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

diabetes = load_diabetes()
"""
print(diabetes)

print(diabetes.DESCR)

print(diabetes.data)

print(diabetes.feature_names)

print(diabetes.target)

#print(diabetes.target_names)
"""
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

df['target'] = diabetes.target
print(df)

sex_data = df[["sex"]]
#print(len(sex_data))
oh_encoder = OneHotEncoder()
encoded_sex_data = oh_encoder.fit_transform(sex_data)
#print(len(encoded_sex_data.toarray()[0:]))
df = df.drop("sex", axis=1)
print(encoded_sex_data.toarray()[1])
#for i in range(len(encoded_sex_data.toarray())):
#    df[i]["sex"] = encoded_sex_data.toarray()[i]
#data = np.reshape(encoded_sex_data.toarray(), (2, -1))
#print(data)
#print(oh_encoder.categories_)


"""
#df = np.c_[encoded_sex_data.toarray(), df]
df['sex'] = encoded_sex_data.toarray()[0:]
print(df)
print(type(encoded_sex_data.toarray()))
"""