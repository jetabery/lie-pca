import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import torch
from torch_implementation import TorusRep

def create_dataset(d=10, n=1000, t=None, freq_range=6, rng=None, noise_factor=.01, shuffle_order=True):
    """
    Creates a synthetic dataset of complex-valued vectors with specified properties.

    Parameters:
    d (int): Dimension of the vectors. Default is 10.
    n (int): Number of vectors to generate. Default is 1000.
    t (array-like, optional): Time points at which to evaluate the vectors. If None, defaults to np.arange(1, n+1)/n.
    freq_range (int): Range of frequencies for the random integers. Default is 6.
    rng (np.random.Generator, optional): Random number generator instance. If None, defaults to np.random.default_rng().
    noise_factor (float): Standard deviation of the Gaussian noise to be added. Default is 0.01.
    shuffle_order (bool): Whether to shuffle the order of the vectors. Default is True.

    Returns:
    np.ndarray: A (d, n) array of complex-valued vectors.
    """
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
    """
    Reorders the dataset based on the angles of the eigenvectors of a kernel matrix.
    Should result in a dataset that is in the same order as the original dataset (modulo the dihedral group).

    Parameters:
    x (numpy.ndarray): Input complex dataset of shape (n_features, n_samples).
    epsilon (float, optional): Scaling parameter for the kernel. If None, it is computed automatically.
    plot_evecs (bool, optional): If True, plots the eigenvectors.
    verbose (bool, optional): If True, prints the value of epsilon.

    Returns:
    tuple: A tuple containing:
        - x (numpy.ndarray): Reordered dataset.
        - weights (numpy.ndarray): Normalized weights.
        - t (numpy.ndarray): Timestamps in the range [0, 1].
    """
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

def get_irreps(x, weights=None, t=None, threshold=0.01, plot_norms=False, log_scale=False, verbose=False, get_vectors=False):
    """
    Computes the irreducible representations (irreps) of the input data.

    Parameters:
    x : numpy.ndarray
        Input data matrix where each column represents a data point.
    weights : numpy.ndarray, optional
        Weights for each data point. If None, uniform weights are used.
    t : numpy.ndarray, optional
        Time points corresponding to each data point. If None, FFT is used.
    threshold : float, optional
        Threshold for detecting significant frequencies. Default is 0.01.
    plot_norms : bool, optional
        If True, plots the norms of the projections to help with finding a good threshold. Default is False.
    log_scale : bool, optional
        If True, uses a logarithmic scale for the norms plot. Default is False.
    verbose : bool, optional
        If True, prints detected frequencies. Default is False.
    get_vectors : bool, optional
        If True, returns normalized projection vectors instead of projection matrices. Default is False.

    Returns:
    dict
        A dictionary where keys are detected frequencies and values are either projection matrices or vectors, depending on the value of get_vectors.
    """
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
    """
    Constructs the orbit of a point `x_start` under the torus representation defined by `irreps_dict`.

    Parameters:
    -----------
    irreps_dict : dict
        A dictionary where keys are frequencies and values are either projection matrices or vectors 
        representing the irreducible representations (irreps) of the torus.
    x_start : numpy.ndarray
        The starting point in the space where the orbit is to be constructed.
    num_points : int, optional
        The number of points to generate along the orbit. Default is 1000.

    Returns:
    --------
    numpy.ndarray
        An array of shape (d, num_points) representing the orbit of the point `x_start` under the torus 
        representation. The array is complex-valued.

    Notes:
    ------
    - If the irreps are projection matrices, the orbit is constructed using classical numpy operations.
    - If the irreps are vectors, the orbit is constructed using the `TorusRep` class.
    """
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
        x0 = torch.from_numpy(x_start)
        model = TorusRep(A, B, omega, x0)
        t = torch.linspace(0, 1, num_points).unsqueeze(-1)
        x_approx = model(t).detach().numpy().T.astype(np.complex128)
        return x_approx

def plot_results(x, x_approx, projection=None, rng=None):
    """
    Plots the original and approximated data in 3D.

    Parameters:
    x : numpy.ndarray
        The original data array. Can be real or complex.
    x_approx : numpy.ndarray
        The approximated data array. Should have the same shape as `x`.
    projection : str, optional
        The type of projection to use for plotting. Options are:
        - None: Use the first three dimensions of the data.
        - 'pca': Use PCA to reduce the data to three dimensions.
        - 'pca_omit1': Use PCA to reduce the data to four dimensions and omit the first component.
        - 'random': Use a random projection to reduce the data to three dimensions.
    rng : numpy.random.Generator, optional
        A random number generator instance for reproducibility. If None, a default RNG will be used.

    Raises:
    ValueError
        If an invalid projection type is provided.

    Returns:
    None
    """
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
    """
    Returns parameters for matrix version of the torus representation.
    Representation takes the form:
    A @ exp(diag(2j * pi * omega * t)) @ B @ x0

    Args:
        irreps_dict (dict): A dictionary where keys are frequencies (or other identifiers) 
                            and values are (d, d) numpy arrays representing vector irreducible 
                            representations (irreps).

    Returns:
        tuple: A tuple containing:
            - A (torch.Tensor): A (d, d) complex tensor representing the torus representation.
            - B (torch.Tensor): The conjugate transpose of A.
            - omega (torch.Tensor): A tensor containing the frequencies (keys of the input dictionary).
    """
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
    x = create_dataset(d=10, n=2000, noise_factor=0.004, rng=rng)
    x, weights, t = reorder_dataset(x)
    
    n = x.shape[1]

    irreps_dict = get_irreps(x=x, weights=weights, t=t, threshold=0.1,\
                    plot_norms=False, verbose=True, get_vectors=True) 

    x_start = x[:, :3].mean(axis=1)
    x_approx = construct_orbit(irreps_dict, x_start, num_points=x.shape[1])

    plot_results(x, x_approx, projection='random', rng=rng)
