import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split


n_samples = 5000

centers = [(-2, -2), (2, 2)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=42)

y[:n_samples // 2] = 0
y[n_samples // 2:] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)
log_reg = linear_model.LogisticRegression()

n_test_samples = X_test.shape[0]

log_reg.fit(X_train, y_train)
y_predict = log_reg.predict(X_test)

# Todo: write to report
# The predictions only have 0 and 1: no

e_count = 0

for i in range(0, n_test_samples):
    if y_predict[i] != y_test[i]:
        e_count += 1
    if y_predict[i] == 0:
        plt.scatter(X_test[i][0], X_test[i][1], c="B")
    else:
        plt.scatter(X_test[i][0], X_test[i][1], c="R")
plt.show()
# Todo: write to report
print ("Number of wrong predictions is: {0}".format(e_count))

