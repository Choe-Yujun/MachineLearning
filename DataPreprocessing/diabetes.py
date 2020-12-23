from sklearn import datasets, linear_model
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

#create diabetes dataframe
diabetes = datasets.load_diabetes()
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

"""
#sex column onehot encoding
sex_data = df[["sex"]]
oh_encoder = OneHotEncoder()
encoded_sex_data = oh_encoder.fit_transform(sex_data)

df = df.drop("sex", axis=1)
print(encoded_sex_data.toarray())
l1 = []
for i in range(len(encoded_sex_data.toarray())):
    l1.append(encoded_sex_data.toarray()[i])
df["sex"]= l1
print(df)
"""
minmax_scaler = MinMaxScaler()
df = minmax_scaler.fit_transform(df)
print(df)

diabetes_X = df[:, np.newaxis, 2]
print(diabetes_X.shape)
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = regr.predict(diabetes_X_test)
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Variance score: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()