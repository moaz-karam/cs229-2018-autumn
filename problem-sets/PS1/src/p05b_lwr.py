import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel

def mse(y_pred, y):
    return np.mean((y_pred - y) ** 2)

def plot(x: np.ndarray, y: np.ndarray, mark, predictor=None, predictor_color='green'):
    plt.plot(x, y, mark, linewidth=2)

    if predictor is not None:
        predictor_arr_x = np.arange(x.min(), x.max(), 0.1).reshape(-1, x.shape[1])
        predictor_arr_y = predictor.predict(predictor_arr_x)
        plt.plot(predictor_arr_x, predictor_arr_y, color=predictor_color)
        
    plt.xlabel('x')
    plt.ylabel('y')

def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    
    lwr = LocallyWeightedLinearRegression(tau)
    lwr.fit(x_train, y_train)

    print("mse:", mse(lwr.predict(x_eval), y_eval))



    # plot(x_train[:, 1], y_train, 'bx')
    plot(x_eval[:, 1], y_eval, 'r.')
    plot(x_eval[:, 1], lwr.predict(x_eval), 'k+')
    plt.show()

    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None


    def get_w(self, x_p):
        """
        Args:
            x_p: Input of shape (n,)
        """
        return np.exp(-np.linalg.norm((self.x - x_p) ** 2, axis=1) / (2 * self.tau ** 2))
        
    def get_theta(self, x_p):
        """
        Args:
            x_p: Inputs of shape (n,)
        Returns:
            theta: Line of shape (n,)
        """
        w = self.get_w(x_p)
        x_w_x_inv: np.ndarray = np.linalg.inv(np.matmul(self.x.T, w.reshape(-1, 1) * self.x))
        x_w_y: np.ndarray = np.matmul(self.x.T, w * self.y)
        return np.matmul(x_w_x_inv, x_w_y)

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict_point(self, x_p):
        return x_p.dot(self.get_theta(x_p))

    def predict(self, x):
        """Make predictions given inputs x.

        Args:p
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        pred = np.array([self.predict_point(x_p) for x_p in x])
        return pred
        # *** END CODE HERE ***


if __name__ == "__main__":
    main(5e-2, "../data/ds5_train.csv", "../data/ds5_valid.csv")
