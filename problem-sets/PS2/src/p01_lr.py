# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    
    probs = 1. / (1 + np.exp(margins))
    # grad = -(1./m) * (X.T.dot(probs * Y))
    # grad = -(1./m) * (X.T.dot(probs * Y)) + 2e-4 * theta
    grad = -(1./m) * (X.T.dot(probs * Y)) + 1e-6 * np.linalg.norm(theta) ** 2
    return grad


def logistic_regression(X, Y, theta=None):
    """Train a logistic regression model."""
    m, n = X.shape
    if theta is None:
        theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return theta

def predict(theta, x):
    return np.matmul(x, theta) > 0

def accuracy(y, y_hat):
    return np.mean(y == y_hat)


def GDA(X: np.ndarray, Y: np.ndarray):
    X = X[:, 1:]
    m, n = X.shape
    Y_copy = Y.copy()
    Y_copy[Y == -1] = 0
    X_0 = X[Y_copy == 0]
    X_1 = X[Y_copy == 1]

    mu_0 = np.sum(X_0, axis=0) / X_0.shape[0]
    mu_1 = np.sum(X_1, axis=0) / X_1.shape[0]

    sigma_mu_0 = np.matmul((X_0 - mu_0).T, X_0 - mu_0)
    sigma_mu_1 = np.matmul((X_1 - mu_1).T, X_1 - mu_1)

    sigma = 1 / m * (sigma_mu_0 + sigma_mu_1)
    sigma_inv = np.linalg.inv(sigma)

    theta = np.matmul(sigma_inv, mu_1 - mu_0)
    mu_0_sigma_mu_0 = np.matmul(mu_0.T, np.matmul(sigma_inv, mu_0))
    mu_1_sigma_mu_1 = np.matmul(mu_1.T, np.matmul(sigma_inv, mu_1))
    phi = 1 / m * X_1.shape[0]
    log_phi = np.log((1 - phi) / phi)
    
    theta_0 = 0.5 * (mu_0_sigma_mu_0 - mu_1_sigma_mu_1 - log_phi)
    theta = np.insert(theta, 0, theta_0)

    return theta

def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)

    theta = logistic_regression(Xa, Ya)
    # theta = GDA(Xa, Ya)
    
    Ya[Ya == -1] = 0
    print('accuracy in set A:', accuracy(Ya, predict(theta, Xa)))

    util.plot(Xa, Ya, theta)
    plt.show()

    print('\n==== Training model on data set B ====')
    theta = logistic_regression(Xb, Yb)
    # theta = GDA(Xb, Yb)
    
    Yb[Yb == -1] = 0    
    print('accuracy in set B:', accuracy(Yb, predict(theta, Xb)))

    util.plot(Xb, Yb, theta)
    plt.show()

if __name__ == '__main__':
    main()
