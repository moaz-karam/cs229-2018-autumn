import numpy as np
import util

from linear_model import LinearModel
from p01b_logreg import LogisticRegression

import matplotlib.pyplot as plt


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)

    # *** START CODE HERE ***
    lg: LogisticRegression = LogisticRegression()
    gda: GDA = GDA()
    lg.fit(x_train, y_train)
    gda.fit(x_train, y_train)
    util.plot(x_val, y_val, [np.insert(lg.theta, 0, 1), gda.get_line()], ["red", "black"])
    print(np.mean(lg.predict(x_val) == y_val))
    print(np.mean(gda.predict(x_val) == y_val))
    plt.show()
    # *** END CODE HERE ***
class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def get_phi(self, y):
        return np.sum(y, axis=0) / y.shape[0]

    def get_m0(self, x0: np.ndarray):
        return np.sum(x0, axis=0) / x0.shape[0]

    def get_m1(self, x1: np.ndarray):
        return np.sum(x1, axis=0) / x1.shape[0]

    def get_sigma(self, x0: np.ndarray, x1: np.ndarray, m0: np.ndarray, m1: np.ndarray):
        m: int = x1.shape[0] + x0.shape[0]
        x0_sum: np.ndarray = np.matmul((x0 - m0).T, (x0 - m0))
        x1_sum: np.ndarray = np.matmul((x1 - m1).T, (x1 - m1))
        return (x0_sum + x1_sum) / m

    def get_theta0(self):
        sigma_inv: np.ndarray = np.linalg.inv(self.sigma)
        m0_sigma_m0: np.ndarray = np.matmul(np.matmul(self.m0.T, sigma_inv), self.m0)
        m1_sigma_m1: np.ndarray = np.matmul(np.matmul(self.m1.T, sigma_inv), self.m1)
        log_phi: np.ndarray = np.log((1 - self.phi) / self.phi)
        return 0.5 * m0_sigma_m0 - 0.5 * m1_sigma_m1 - log_phi

    def get_theta(self):
        sigma_inv: np.ndarray = np.linalg.inv(self.sigma)
        return np.matmul(sigma_inv, self.m1 - self.m0)
    
    def p_y_given_x(self, x: np.ndarray):
        return 1 / (1 + np.exp(-np.matmul(x, self.theta) - self.theta0))

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        x0 = x[y == 0]
        x1 = x[y == 1]
        self.phi: float = self.get_phi(y)
        self.m0: np.ndarray = self.get_m0(x0)
        self.m1: np.ndarray = self.get_m1(x1)
        self.sigma: np.ndarray = self.get_sigma(x0, x1, self.m0, self.m1)
        self.theta0: np.ndarray = self.get_theta0()
        self.theta: np.ndarray = self.get_theta()
        # *** END CODE HERE ***

    def predict(self, x: np.ndarray):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # return self.p_y_given(x) >= 0.5 OR
        return (np.matmul(x, self.theta) + self.theta0) >= 0
        # *** END CODE HERE

    def get_line(self):
        return np.insert(self.theta, 0, self.theta0)
