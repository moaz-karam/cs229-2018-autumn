
import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    # The line below is the original one from Stanford. It does not include the intercept, but this should be added.
    # x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    poisson = PoissonRegression()
    poisson.fit(x_train, y_train, lr)
    print(x_train.shape)
    print(y_train.shape)
    print(poisson.theta)
    print("poisson mean difference (eval):", np.mean(poisson.predict(x_eval) - y_eval))
    print("poisson mean difference (train):", np.mean(poisson.predict(x_train) - y_train))
    def plot(y_label, y_pred, title):
        plt.plot(y_label, 'go', label='label')
        plt.plot(y_pred, 'rx', label='prediction')
        plt.suptitle(title, fontsize=12)
        plt.legend(loc='upper left')

    plot(y_eval, poisson.predict(x_eval), 'eval')
    plt.show()
    plot(y_train, poisson.predict(x_train), 'train')
    plt.show()
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def h(self, x: np.ndarray) -> np.ndarray:
        
        """
        Args:
            x: Training example inputs. Shape (m, n)

        Returns:
             Shape(m,)
        """
        return np.exp(np.matmul(x, self.theta))

    def get_new_theta(self, x: np.ndarray, y: np.ndarray):
        h_x = self.h(x)
        return self.theta + self.lr * np.matmul(x.T, (y - h_x)) / x.shape[0]
    
    def gradient(self, x):
        return np.matmul(x.T, self.h(x))

    def fit(self, x, y, lr):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x[0].shape)

        self.lr = lr

        new_theta = self.get_new_theta(x, y)
        while np.linalg.norm(new_theta - self.theta) > self.eps:
            self.theta = new_theta
            new_theta = self.get_new_theta(x, y)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return self.h(x)
        # *** END CODE HERE ***

if __name__ == "__main__":
    main(1e-7, "../data/ds4_train.csv", "../data/ds4_valid.csv", "")
