import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################


def my_fit(X_train, y0_train, y1_train):
    # Transform the challenges
    transformed_challenges = my_map(X_train)

    # Train two linear models
    model0 = LogisticRegression(C=10.0, tol=1e-5, solver='liblinear')
    model0.fit(transformed_challenges, y0_train)

    model1 = LogisticRegression(C=10.0, tol=1e-5, solver='liblinear')
    model1.fit(transformed_challenges, y1_train)

    # Extract weights and biases
    W0 = model0.coef_[0]
    b0 = model0.intercept_[0]
    W1 = model1.coef_[0]
    b1 = model1.intercept_[0]

    return W0, b0, W1, b1
################################
#  Non Editable Region Ending  #
################################

################################
# Non Editable Region Starting #
################################


def my_map(X):
    # Using the Khatri-Rao product to transform the feature vector
    transformed_features = khatri_rao(X.T, X.T).T
    return transformed_features
################################
#  Non Editable Region Ending  #
################################

# Ensure the file has the correct structure and logic.
