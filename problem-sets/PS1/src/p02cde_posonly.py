import numpy as np
import util
import matplotlib.pyplot as plt

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    x_train, t_train = util.load_dataset(train_path, label_col="t", add_intercept=True)
    x_valid, t_valid = util.load_dataset(valid_path, label_col="t", add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col="t", add_intercept=True)
    
    _, y_train = util.load_dataset(train_path, label_col="y", add_intercept=True)
    _, y_valid = util.load_dataset(valid_path, label_col="y", add_intercept=True)
    _, y_test = util.load_dataset(test_path, label_col="y", add_intercept=True)
    
    # using the t lables only
    lg_t: LogisticRegression = LogisticRegression()
    lg_t.fit(x_train, t_train)
    print("logistic regression accuracy for t:", np.mean(lg_t.predict(x_test) == t_test))
    t_only_theta = lg_t.theta

    #using the y labels only
    lg_y: LogisticRegression = LogisticRegression()
    lg_y.fit(x_train, y_train)
    y_only_theta = lg_y.theta

    
    print("logistic regression accuracy for y:", np.mean(lg_y.predict(x_test)  == y_test))

    x_valid_positive: np.ndarray = x_valid[y_valid == 1]
    alpha: float = np.mean(lg_y.h(x_valid_positive))
    def scaled_predict(self: LogisticRegression, x, alpha):
        return  (self.h(x) / alpha) >= 0.5
    
    scaled_predict
    print((scaled_predict(lg_y, x_test, alpha) == t_test).mean())

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE

if __name__ == "__main__":
    main("../data/ds3_train.csv", "../data/ds3_valid.csv", "../data/ds3_test.csv", "")
