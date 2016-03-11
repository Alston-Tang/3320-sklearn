import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


X_train = [[6], [8], [10], [14], [18]]
y_train = [[7.5], [9.1], [13.2], [17.5], [19.3]]

X_test = [[6], [8], [11], [16]]
y_test = [[8.3], [12.5], [15.4], [18.6]]

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Todo: write to report
print ("y1= {0} + {1} x".format(lr_model.intercept_[0], lr_model.coef_[0][0]))
xx = np.linspace(0, 26, 100)
yy = lr_model.predict(xx.reshape(xx.shape[0], 1))
lr_score = lr_model.score(X_test, y_test)
# Todo: write to report
print ("Linear regression (order 1) model score is: {0}".format(lr_score))
plt.plot(xx, yy)
plt.plot(X_test, y_test, "o")
plt.title("Linear regression (order 1) result")
plt.show()


poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

lr_5_model = LinearRegression()
lr_5_model.fit(X_train_poly, y_train)
# Todo: write to report
print ("y2= {0} + {1} x + {2} x*x + {3} x*x*x + {4} x*x*x*x +{5} x*x*x*x*x".
       format(lr_5_model.intercept_[0], lr_5_model.coef_[0][0], lr_5_model.coef_[0][1], lr_5_model.coef_[0][2],
              lr_5_model.coef_[0][3], lr_5_model.coef_[0][4]))

xx_poly = poly.transform(xx.reshape(xx.shape[0], 1))
yy_poly = lr_5_model.predict(xx_poly)

print ("Linear regression (order 5) score is: {0}".format(lr_5_model.score(X_test_poly, y_test)))

plt.plot(xx, yy_poly)
plt.plot(X_test, y_test, "o")
plt.ylim([0, 30])
plt.title("Linear regression (order 5) result")
plt.show()


ridge_model = Ridge(alpha=1, normalize=False)
ridge_model.fit(X_train_poly, y_train)
yy_ridge = ridge_model.predict(xx_poly)

# Todo: write to report
print ("Ridge regression (order 5) score is: {0}".format(ridge_model.score(X_test_poly, y_test)))
print ("y2= {0} + {1} x + {2} x*x + {3} x*x*x + {4} x*x*x*x +{5} x*x*x*x*x".
       format(ridge_model.intercept_[0], ridge_model.coef_[0][0], ridge_model.coef_[0][1], ridge_model.coef_[0][2],
              ridge_model.coef_[0][3], ridge_model.coef_[0][4]))

plt.plot(xx, yy_ridge)
plt.plot(X_test, y_test, "o")
plt.ylim([0, 30])
plt.title("Ridge regression (order 5) result")
plt.show()

# Compare
# 1. The model with the highest score is: Ridge model (order 5)
# 2. Ridge model can prevent over-fitting: yes
# 3. Ridge model is nearly equivalent to LR model (order 5) if alpha=0: yes
# 4. A larger alpha results in a larger coefficient for x*x*x*x*x: no
# 5. Best score is achieved at order 2

max_score = None
max_ridge_score = None
best_order = None
best_ridge_order = None

for i in range(1, 100):
    poly = PolynomialFeatures(degree=i)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    lr_model = LinearRegression()
    lr_model.fit(X_train_poly, y_train)
    score = lr_model.score(X_test_poly, y_test)
    ridge_model = Ridge(alpha=1)
    ridge_model.fit(X_train_poly, y_train)
    ridge_score = ridge_model.score(X_test_poly, y_test)

    if score > max_score or max_score is None:
        max_score = score
        best_order = i
    if ridge_score > max_ridge_score or max_ridge_score is None:
        max_ridge_score = ridge_score
        best_ridge_order = i

print best_order
print max_score
print best_ridge_order
print max_ridge_score


