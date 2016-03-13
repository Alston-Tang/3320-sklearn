import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
import seaborn

seaborn.set()

name = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price"
]

df = pd.read_csv('imports-85.data',
                 header=None,
                 names=name,
                 na_values="?"
                 )

df = df.dropna(axis=0)

n_features = df.shape[1]
n_samples = df.shape[0]


scaler = StandardScaler()
samples_eng_size = scaler.fit_transform(df.iloc[:, 16].astype(float).reshape(-1, 1))
samples_peak_rpm = scaler.fit_transform(df.iloc[:, 22].reshape(-1, 1))
samples_price = scaler.fit_transform(df.iloc[:, 25].reshape(-1, 1))

x = np.append(samples_eng_size, samples_peak_rpm, axis=1)

unit_feature = np.ones((n_samples, 1))

u_x = np.append(unit_feature, x, axis=1)
u_x_t = np.transpose(u_x)


theta = np.dot(np.dot(np.linalg.inv(np.dot(u_x_t, u_x)), u_x_t), samples_price)

print ("Parameter theta calculated by normal equation: [{0}, {1}, {2}]".format(theta[0][0], theta[1][0], theta[2][0]))

clf = linear_model.SGDRegressor(loss="squared_loss")
clf.fit(x, np.ravel(samples_price))

print ("Parameter theta calculated by SGD: [{0}, {1}, {2}]".
       format(clf.intercept_[0], clf.coef_[0], clf.coef_[1]))



