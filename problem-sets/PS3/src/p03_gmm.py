import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 5  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)
LABELED_PER_CLUSTER = 5

def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None
    m, n = x.shape

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group

    # shuffling
    rng = np.random.default_rng()
    rng.shuffle(x)
    elements_per_cluster = m / K
    mu = np.zeros(shape=(K, n))
    sigma = np.zeros(shape=(K, n, n))
    
    for i in range(K):
        current_cluster = x[i * int(elements_per_cluster): (i + 1) * int(elements_per_cluster)]
        mu[i] = np.mean(current_cluster, axis=0)
        sigma[i] = np.cov(current_cluster, rowvar=False)

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full(shape=(K,), fill_value=1 / K)
    
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)

    w = np.full(shape=(m, K), fill_value=1 / K)
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    m, _ = x.shape
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)

def normal(x_i, mu, sigma):
    sign, log_det = np.linalg.slogdet(sigma)
    root_sigma_det = sign * np.exp(log_det * 0.5)
    # root_sigma_det = np.sqrt(np.linalg.det(sigma))
    sigma_inv = np.linalg.inv(sigma)
    x_i = x_i.reshape(1, x_i.shape[0], 1)
    d = x_i.shape[0]

    mu = mu.reshape(mu.shape[0], mu.shape[1], 1)
    mu_trans = np.transpose((x_i - mu), axes=(0, 2, 1))

    factor = 1 / (np.sqrt((2 * np.pi) ** d) * root_sigma_det)
    
    return (
        factor *
        np.exp(-0.5 *
               np.matmul(mu_trans, (np.matmul(sigma_inv, x_i - mu)))).reshape(-1)
    )


def em_e_step(x_i, phi, mu, sigma):
    """
    applies e_step for a single row 'i'
    returning the i-th row for the w_i
    """
    p_x_z = normal(x_i, mu, sigma) * phi
    p_x = np.sum(p_x_z)

    return p_x_z / p_x

def em_m_step(x, w, mu):
    m, n = x.shape
    w_sum = np.sum(w, axis=0)
    phi = w_sum / m
    new_mu = np.divide(
        w.T.dot(x),
        w_sum.reshape(-1, 1),
        np.zeros_like(w.T.dot(x)),
        where=w_sum.reshape(-1, 1)!=0)
    
    x_minus_mu = x - mu.reshape((mu.shape[0], 1, mu.shape[1]))
    x_minus_mu_T = np.transpose(x_minus_mu, axes=(0, 2, 1))

    # multiply one by the w
    w_x_minus_mu = x_minus_mu * w.T.reshape(w.T.shape + (1, ))

    x_minus_mu_squared = np.matmul(x_minus_mu_T, w_x_minus_mu)

    sigma = np.divide(
        x_minus_mu_squared,
        w_sum.reshape(w_sum.shape + (1, 1)),
        x_minus_mu_squared,
        where=w_sum.reshape(w_sum.shape + (1, 1))!=0)
    
    return (phi, new_mu, sigma)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 100

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        w = np.apply_along_axis(em_e_step, 1, x, phi, mu, sigma)
        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi, mu, sigma = em_m_step(x, w, mu)
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        p_x = np.matmul(w, phi)
        prev_ll = ll
        ll = np.sum(np.log(p_x))
        # *** END CODE HERE ***

    return w

def semi_supervised_e_step(x_i, phi, mu, sigma):
    return em_e_step(x_i, phi, mu, sigma)

def semi_supervised_m_step(x, x_tilde, z, w, mu, alpha):
    m, n = x.shape
    m_tilde, _ = x_tilde.shape
    w_sum = np.sum(w, axis=0)
    common_deno = w_sum + alpha * LABELED_PER_CLUSTER
    clustered_x_tilde = np.zeros(shape=(K, int(m_tilde / K), n))

    for i in range(K):
        clustered_x_tilde[i] = x_tilde[z.reshape(-1) == i]

    summed_clustered_x_tilde = np.sum(clustered_x_tilde, axis=1)
    new_phi = common_deno / (m + alpha * m_tilde)
    new_mu = (
        (w.T.dot(x) + alpha * summed_clustered_x_tilde)
    ) / common_deno.reshape(-1, 1)

    x_minus_mu = x - mu.reshape((mu.shape[0], 1, mu.shape[1]))
    x_minus_mu_T = np.transpose(x_minus_mu, axes=(0, 2, 1))

    x_tilde_minus_mu = clustered_x_tilde - mu.reshape((mu.shape[0], 1, mu.shape[1]))
    x_tilde_minus_mu_T = np.transpose(x_tilde_minus_mu, axes=(0, 2, 1))

    w_x_minus_mu = x_minus_mu * w.T.reshape(w.T.shape + (1, ))
    w_x_minus_mu_squared = np.matmul(x_minus_mu_T, w_x_minus_mu)

    x_tilde_minus_mu_squared = np.matmul(x_tilde_minus_mu_T, x_tilde_minus_mu)

    new_sigma = (
        (w_x_minus_mu_squared + alpha * x_tilde_minus_mu_squared) /
        (common_deno).reshape(common_deno.shape + (1, 1))
    )

    return (new_phi, new_mu, new_sigma)

def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        w = np.apply_along_axis(semi_supervised_e_step, 1, x, phi, mu, sigma)
        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi, mu, sigma = semi_supervised_m_step(x, x_tilde, z, w, mu, alpha)
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        p_x = np.matmul(w, phi)
        prev_ll = ll
        ll = np.sum(np.log(p_x))
       
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(True, trial_num=t)
        # *** END CODE HERE ***
