import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import torch
from torch_implementation import TorusRep

def create_dataset(d=10, n=1000, t=None, \
        freq_range=6, rng=None, noise_factor=.01, shuffle_order=True):
    if t is None:
        t = np.arange(1, n+1)/n
    if rng is None:
        rng = np.random.default_rng()

    Q = unitary_group(dim=d, seed=rng).rvs()  # random unitary matrix
    a = rng.integers(low=-freq_range, high=freq_range, endpoint=True, size=d)  # frequencies
    v = rng.normal(0, 1, d) + 1j*rng.normal(0, 1, d)
    v = v / np.linalg.norm(v)
    x = np.zeros((d, n), dtype=complex)
    for j in range(n):
        x[:, j] = Q @ np.diag(np.exp(2 * np.pi * 1j * a * t[j])) @ Q.T.conj() @ v
    # print(np.max(x.imag))
    noise = (rng.standard_normal((d, n)) + 1j * rng.standard_normal((d, n))) * noise_factor
    x = x + noise
    if shuffle_order:
        x = x[:, rng.permutation(n)] 
    return x

def reorder_dataset(x, epsilon=None, plot_evecs=False, verbose=False):
    dists = squareform(pdist((np.vstack((x.real, x.imag)).T)))
    if epsilon is None:
        # epsilon = np.min(dists + np.eye(dists.shape[1])) ** 2
        epsilon = (dists + np.eye(dists.shape[1])).min(axis=0).mean() ** 2
    if verbose:
        print('epsilon:', epsilon)
    K = np.exp(-1 * dists**2 / epsilon)
    weights = np.ones(x.shape[1])
    for _ in range(10):  # Sinkhorn
        inv_col_sums = 1 / np.sqrt(K.sum(axis=0))
        weights = weights * inv_col_sums
        weights = weights / np.sum(weights)  # Normalize weights
        K = inv_col_sums * K * inv_col_sums[:, None]
        K = (K + K.T)/2
    weights = weights / np.sum(weights)
    _, evecs = np.linalg.eigh(K)
    angles = np.angle(evecs[:, -2] + 1j*evecs[:, -3])
    if plot_evecs:
        plt.plot(evecs[:, -2], evecs[:, -3], 'o')
        plt.show()
    idx = np.argsort(angles)
    t = (angles[idx] + np.pi) / (2 * np.pi) # timestamps in [0, 1]
    x = x[:, idx]
    weights = weights[idx]
    return x, weights, t

def get_irreps(x, weights=None, t=None, threshold=0.01, plot_norms=False, \
               log_scale=False, verbose=False, get_vectors=False):
    ''' Returns a dictionary.
    The keys are integers, corresponding to frequencies.
    The values are matrices or vectors (depending on get_projections), corresponding to projections onto irreps
    '''
    if t is None:
        assert weights is None
        n = x.shape[1]
        fft_x = fft(x)
        detected_freqs = fftfreq(n, 1/n)[np.where(np.linalg.norm(fft_x, axis=0) > threshold*n)].astype(np.int64)
        if verbose:
            print('Detected Frequencies:', detected_freqs)
        P = {}
        for freq in detected_freqs:
            Pbx = fft_x[:, freq]
            if get_vectors:
                P[freq] = Pbx / np.linalg.norm(Pbx)
            else:
                P[freq] = (np.linalg.norm(Pbx) ** -2) * np.outer(Pbx, Pbx.conj())
        return P
    else:
        n = x.shape[1]
        if weights is None:
            weights = np.ones(n) / n
        P = {}
        norms = []
        for freq in range(-n//2, n//2 + 1):
            Pbx = np.sum(x * np.exp(-2j * np.pi * freq * t) * weights, axis=1)
            norms.append(np.linalg.norm(Pbx))
            if np.linalg.norm(Pbx) > threshold:
                if get_vectors:
                    P[freq] = Pbx / np.linalg.norm(Pbx)
                else:
                    P[freq] = (np.linalg.norm(Pbx) ** -2) * np.outer(Pbx, Pbx.conj())
        if verbose:
            print('Detected Frequencies:', P.keys())
        if plot_norms: # to help with finding a good threshold
            if log_scale:
                norms = np.log10(norms)
            plt.plot(range(-n//2, n//2 + 1), norms)
            plt.xlabel('Frequency')
            if log_scale:
                plt.ylabel('Log Norm (base 10)')
            else:
                plt.ylabel('Norm')
            plt.title('Norms of Projections')
            plt.show()
        return P

def construct_orbit(irreps_dict, x_start, num_points=1000):
    '''
    Constructs the orbit of a point x_start under the torus representation defined by irreps_dict.
    If the irreps are projection matrices, constructs the orbit using classical numpy operations.
    If the irreps are vectors, constructs the orbit using the TorusRep class.
    '''
    irreps_shape = list(irreps_dict.values())[0].shape
    from_proj_matrices = len(irreps_shape) == 2
    if from_proj_matrices:
        d = x_start.shape[0]
        n = num_points
        t = np.arange(1, n + 1) / n
        x_approx = np.zeros((d, n), dtype=complex)
        for j in range(n):
            Gj = np.eye(d, dtype=complex)
            for freq in irreps_dict.keys():
                Gj += irreps_dict[freq] * (np.exp(2j * np.pi * freq * t[j]) - 1)
            x_approx[:, j] = Gj @ x_start
        return x_approx
    else:
        A, B, omega = get_params_for_TorusRep(irreps_dict)
        x0 = torch.from_numpy(x[:, 0])
        model = TorusRep(A, B, omega, x0)
        t = torch.linspace(0, 1, num_points).unsqueeze(-1)
        x_approx = model(t).detach().numpy().T.astype(np.complex128)
        return x_approx

def plot_results(x, x_approx, projection=None, rng=None):
    is_complex = x.dtype == complex
    if rng is None:
        rng = np.random.default_rng()
    if is_complex:
        x_real = np.vstack([x.real, x.imag]).T  # Shape becomes (n, 2*d)
        x_approx_real = np.vstack([x_approx.real, x_approx.imag]).T
    else:
        x_real = x.T
        x_approx_real = x_approx.T

    if projection is None:
        x_plot = x_real[:, :3]
        x_approx_plot = x_approx_real[:, :3]
    elif projection == 'pca':
        pca = PCA(n_components=3)
        x_plot = pca.fit_transform(x_real)  # Shape will be (n, 3)
        x_approx_plot = pca.transform(x_approx_real)
    elif projection == 'pca_omit1':
        pca = PCA(n_components=4)
        x_plot = pca.fit_transform(x_real)[:, 1:]  # Shape will be (n, 3)
        x_approx_plot = pca.transform(x_approx_real)[:, 1:]
    elif projection == 'random':
        if is_complex:
            proj_mat, _ = np.linalg.qr(rng.normal(0, 1, (2*x.shape[0], 3)))
        else:
            proj_mat, _ = np.linalg.qr(rng.normal(0, 1, (x.shape[0], 3)))
        x_plot = x_real @ proj_mat
        x_approx_plot = x_approx_real @ proj_mat
    else:
        raise ValueError('enter a valid projection: pca, random, or None')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_plot[:, 0], x_plot[:, 1], x_plot[:, 2], label='Original Data', s=1)
    ax.plot(x_approx_plot[:, 0], x_approx_plot[:, 1], x_approx_plot[:, 2], \
            label='Approximated', color='red')
    ax.legend()
    ax.set_axis_off()
    plt.show()

def get_params_for_TorusRep(irreps_dict):
    ''' Returns a matrix representation of the torus representation. 
    Accepts a dictionary of (vector) irreps. 
    Returns a (d,d) pytorch tensor'''
    d = list(irreps_dict.values())[0].shape[0]
    A = np.zeros((d, d), dtype=np.complex128)
    for k, Pbx in enumerate(irreps_dict.values()):
        Pbx = Pbx[:, None]
        basis_vector = np.zeros(d, dtype=np.complex128)[:, None]
        basis_vector[k, 0] = 1
        A += Pbx @ basis_vector.T
    A = torch.from_numpy(A)
    B = A.conj().T
    omega = torch.tensor([freq for freq in irreps_dict.keys()])
    return A, B, omega


if __name__=='__main__':
    rng = np.random.default_rng()
    x = create_dataset(d=10, n=2000, noise_factor=0.002, rng=rng)
    x, weights, t = reorder_dataset(x)
    
    n = x.shape[1]

    irreps_dict = get_irreps(x=x, weights=weights, t=t, threshold=0.1,\
                    plot_norms=False, verbose=True, get_vectors=True) 

    x_start = x[:, 0]
    x_approx = construct_orbit(irreps_dict, x_start, num_points=x.shape[1])
    print(x_approx.shape)

    plot_results(x, x_approx, projection='random', rng=rng)
