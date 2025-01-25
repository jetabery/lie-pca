import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

def create_dataset(d=10, n=1000, t=None, \
        freq_range=6, rng=None, noise_factor=.01, shuffle_order=True):
    if t is None:
        t = np.arange(1, n+1)/n
    if rng is None:
        rng = np.random.default_rng()

    # Create Dataset
    Q = unitary_group(dim=d, seed=rng).rvs()  # random untary matrix
    a = rng.integers(low=-freq_range, high=freq_range, endpoint=True, size=d)  # frequencies
    v = rng.normal(0, 1, d) + 1j*rng.normal(0, 1, d)
    v = v / np.linalg.norm(v)
    x = np.zeros((d, n), dtype=complex)
    for j in range(n):
        x[:, j] = Q @ np.diag(np.exp(2 * np.pi * 1j * a * t[j])) @ Q.T.conj() @ v
    # print(np.max(x.imag))
    x = x + (np.random.randn(d, n) + 1j * np.random.randn(d, n)) * noise_factor  
    if shuffle_order:
        x = x[:, rng.permutation(n)] 
    return x

def reorder_dataset(x):
    dists = squareform(pdist((np.vstack((x.real, x.imag)).T)))
    epsilon = np.min(dists + np.eye(dists.shape[1])) ** 2
    K = np.exp(-1 * dists**2 / epsilon)
    for k in range(10):  # Sinkhorn
        inv_col_sums = 1 / np.sqrt(K.sum(axis=0))
        K = inv_col_sums * K * inv_col_sums[:, None]
        K = (K + K.T)/2
    evals, evecs = np.linalg.eigh(K)
    angles = np.angle(evecs[:, -2] + 1j*evecs[:, -3])
    idx = np.argsort(angles)
    x = x[:, idx]
    return x

def get_irreps(x):
    ''' Returns a dictionary.
    The keys are integers, corresponding to frequencies.
    The values are matrices, corresponding to projections onto irreps
    '''
    n = x.shape[1]
    fft_x = fft(x)
    detected_freqs = fftfreq(n, 1/n)[np.where(np.linalg.norm(fft_x, axis=0) > 0.01*n)].astype(int)
    P = {}
    for freq in detected_freqs:
        Pbx = fft_x[:, freq]
        P[freq] = (np.linalg.norm(Pbx) ** -2) * np.outer(Pbx, Pbx.conj())
    return P

def construct_orbit(P, x_start, num_points=1000):
    d = x_start.shape[0]
    n = num_points
    t = np.arange(1, n + 1) / n
    x_approx = np.zeros((d, n), dtype=complex)
    for j in range(n):
        Gj = np.eye(d, dtype=complex)
        for freq in P:
            Gj += P[freq] * (np.exp(2j * np.pi * freq * t[j]) - 1)
        x_approx[:, j] = Gj @ x_start
    return x_approx

def plot_results(x, x_approx, projection=None):
    x_real = np.vstack([x.real, x.imag]).T  # Shape becomes (n, 2*d)
    x_approx_real = np.vstack([x_approx.real, x_approx.imag]).T

    if projection is None:
        x_plot = x_real[:, :3]
        x_approx_plot = x_approx_real[:, :3]
    elif projection == 'pca':
        pca = PCA(n_components=3)
        x_plot = pca.fit_transform(x_real)  # Shape will be (n, 3)
        x_approx_plot = pca.transform(x_approx_real)
    elif projection == 'random':
        proj_mat, _ = np.linalg.qr(rng.normal(0, 1, (2*x.shape[0], 3)))
        x_plot = x_real @ proj_mat
        x_approx_plot = x_approx_real @ proj_mat
    else:
        raise ValueError('enter a valid projection: pca, random, or None')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_plot[:, 0], x_plot[:, 1], x_plot[:, 2], label='Original', s=1)
    ax.plot(x_approx_plot[:, 0], x_approx_plot[:, 1], x_approx_plot[:, 2], label='Approximated', color='red')
    ax.legend()
    plt.show()

if __name__=='__main__':
    rng = np.random.default_rng()
    x = create_dataset(d=10, n=1000, noise_factor=0.01,  rng=rng) # unordered
    x = reorder_dataset(x) # so that fft makes sense
    P = get_irreps(x)  # get frequencies and corresponding irreps
    x_start = np.sum(x[:, :5], axis=1) / 5
    x_approx = construct_orbit(P, x_start, num_points=x.shape[1])

    plot_results(x, x_approx, projection='random')