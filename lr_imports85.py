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

slice_pos = df.shape[0] * 4 / 5

train = df[0:slice_pos]
test = df[slice_pos: -1]

train_x = train.iloc[:, 16].reshape(-1, 1)
train_y = train.iloc[:, 25].reshape(-1, 1)
test_x = test.iloc[:, 16].reshape(-1, 1)
test_y = test.iloc[:, 25].reshape(-1, 1)

model = linear_model.LinearRegression()
model.fit(train_x, train_y)

predict_y = model.predict(test_x)

plt.plot(test_x, test_y, "bo")
plt.plot(test_x, predict_y, "ro")
plt.xlabel("Engine size")
plt.ylabel("Price")
plt.title("Linear regression on clean data")
plt.show()

price_eng_175 = model.predict(175)
print ("Price prediction for engine size equals to 175 is: {0}".format(price_eng_175[0][0]))

x_scaler = StandardScaler()
train_x_scaled = x_scaler.fit_transform(train_x)
test_x_scaled = x_scaler.fit_transform(test_x)
train_y_scaled = x_scaler.fit_transform(train_y)
test_y_scaled = x_scaler.fit_transform(test_y)

model_scaled = linear_model.LinearRegression()
model_scaled.fit(train_x_scaled, train_y_scaled)

predict_y_scaled = model_scaled.predict(test_x_scaled)

plt.plot(test_x_scaled, test_y_scaled, "bo")
plt.plot(test_x_scaled, predict_y_scaled, "ro")
plt.xlabel("Standardized Engine size")
plt.ylabel("Standardized Price")
plt.title("Linear regression on standardized data"
          "")
plt.show()

