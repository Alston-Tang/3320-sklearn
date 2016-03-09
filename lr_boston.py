import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn; seaborn.set()

# Load the diabetes dataset
boston = datasets.load_boston()

n_samples = boston.data.shape[0]
n_features = boston.data.shape[1]

print ("Number of features in the Boston dataset is: {0}".format(n_features))
print ("Number of samples in the Boston dataset is: {0}".format(n_samples))



# which feature
best_feature_name = None
best_feature_score = 0

for i_feature in range(0, n_features):

    # Get the feature name
    feature_name = boston.feature_names[i_feature]

    # Use only one feature
    diabetes_X = boston.data[:, np.newaxis, i_feature]

    # Split the data into training/testing sets
    boston_X_train = diabetes_X[:-20]
    boston_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    boston_y_train = boston.target[:-20]
    boston_y_test = boston.target[-20:]

    # Create linear regression object
    model = linear_model.LinearRegression()

    # Train the model using the training sets
    model.fit(boston_X_train, boston_y_train)

    # Explained variance score: score=1 is perfect prediction
    model_score = model.score(boston_X_test, boston_y_test)

    if model_score > best_feature_score:
        best_feature_name = feature_name
        best_feature_score = model_score


print ("Best fitted feature name is: {0}".format(best_feature_name))
print ("Best fitted model score is: {0}".format(best_feature_score))