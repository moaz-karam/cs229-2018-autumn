import numpy as np
import util
import pprint as pp
import matplotlib.pyplot as plt
import math

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression()
    model.fit(x_train, y_train)
    print("The accuracy on validation set is: ", np.mean(model.predict(x_val) == y_val))
    with open(pred_path, "w") as file:
        file.write(str(model.predict(x_val)))
    # *** END CODE HERE ***

class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """


    def h(self, x):
        return self.sigmoid(x.dot(self.theta))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def l_dash(self, X, Y):
        return -1 / X.shape[0] * np.dot(X.T, (Y - self.h(X)))

    def l_double_dash(self, X):
        factor = (self.h(X) * (1 - self.h(X))).reshape(-1, 1)
        X_factor = X * factor
        return np.matmul(X.T, X_factor) / X.shape[0]
	
    def get_theta_1(self, X, Y):
        hessian_inv = np.linalg.inv(self.l_double_dash(X))
        return self.theta - (np.matmul(hessian_inv, self.l_dash(X, Y)))
    
    def fit(self, X, Y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(X[0].shape)
        theta_1 = self.get_theta_1(X, Y)
        while np.linalg.norm(theta_1 - self.theta, 1) > self.eps:
            self.theta = theta_1
            theta_1 = self.get_theta_1(X, Y)
        self.theta = theta_1
        # *** END CODE HERE ***

    def predict(self, X):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return np.matmul(X, self.theta) >= 0
        # *** END CODE HERE ***

    def reset(self):
        self.theta = None

# if __name__ == "__main__":
#     main("../data/ds1_train.csv", "../data/ds1_valid.csv", "")
