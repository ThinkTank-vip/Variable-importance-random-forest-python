import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
#clf.fit(predictors.values, outcome.values.ravel())
dataset = pd.read_excel('File Path')
# fit an Extra Trees model to the data
ylog1p_train = dataset['Target Variable']
df_train = dataset.drop(['Drop ids and useless columns'], axis=1)

df_train_dummy = pd.get_dummies(df_train)
clf.fit(df_train_dummy.values, ylog1p_train.values.ravel())

importance = clf.feature_importances_
importance = pd.DataFrame(importance, index=df_train_dummy.columns, 
                          columns=["Importance"])

importance["Std"] = np.std([tree.feature_importances_
                            for tree in clf.estimators_], axis=0)

x = range(importance.shape[0])
y = importance.ix[:, 0]
yerr = importance.ix[:, 1]

importance.to_csv('path')

plt.bar(x, y, align="center")

plt.show()