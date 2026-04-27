import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# Spectral model utilities
# ------------------------------------------------------------------
def gaussian(x, centre, width, amp):
    """One-dimensional Gaussian."""
    return amp * np.exp(-0.5 * ((x - centre) / width)**2)

def generate_gaussian_basis(wavelengths, params):
    """
    Evaluate a set of Gaussian basis functions.

    Parameters
    ----------
    wavelengths : ndarray, shape (N,)
        Wavelength grid.
    params : list of tuples
        Each tuple is (centre, width, amp).

    Returns
    -------
    basis : ndarray, shape (N, M)
        Matrix of basis functions evaluated on the grid.
    """
    M = len(params)
    basis = np.zeros((len(wavelengths), M))
    for i, (c, w, a) in enumerate(params):
        basis[:, i] = gaussian(wavelengths, c, w, a)
    return basis

def generate_synthetic_spectra(n_samples, wavelengths, params, rng=None):
    """
    Generate synthetic spectra as linear combinations of Gaussian bases.

    Parameters
    ----------
    n_samples : int
        Number of spectra.
    wavelengths : ndarray, shape (N,)
        Wavelength grid.
    params : list of tuples
        Parameters for the Gaussian basis functions.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    spectra : ndarray, shape (n_samples, N)
        Synthetic spectra.
    coeffs : ndarray, shape (n_samples, M)
        Coefficients for the basis functions.
    """
    if rng is None:
        rng = np.random.default_rng()
    basis = generate_gaussian_basis(wavelengths, params)
    M = basis.shape[1]
    coeffs = rng.uniform(0.5, 1.5, size=(n_samples, M))
    spectra = coeffs @ basis.T
    return spectra, coeffs

# ------------------------------------------------------------------
# Photometry utilities
# ------------------------------------------------------------------
def make_filter(wavelengths, centre, width):
    """
    Construct a top-hat filter transmission.

    Parameters
    ----------
    wavelengths : ndarray, shape (N,)
        Wavelength grid.
    centre : float
        Filter centre wavelength.
    width : float
        Full-width of the filter.

    Returns
    -------
    transmission : ndarray, shape (N,)
        Filter transmission curve.
    """
    low, high = centre - 0.5 * width, centre + 0.5 * width
    return np.where((wavelengths >= low) & (wavelengths <= high), 1.0, 0.0)

def generate_photometry(spectra, wavelengths, filter_list):
    """
    Compute synthetic photometric fluxes by integrating spectra through filters.

    Parameters
    ----------
    spectra : ndarray, shape (n_samples, N)
        Input spectra.
    wavelengths : ndarray, shape (N,)
        Wavelength grid.
    filter_list : list of dict
        Each dict has keys 'name', 'centre', 'width'.

    Returns
    -------
    fluxes : ndarray, shape (n_samples, K)
        Integrated fluxes for each filter.
    """
    dlam = wavelengths[1] - wavelengths[0]
    K = len(filter_list)
    n_samples = spectra.shape[0]
    fluxes = np.zeros((n_samples, K))
    for j, flt in enumerate(filter_list):
        trans = make_filter(wavelengths, flt['centre'], flt['width'])
        fluxes[:, j] = np.sum(spectra * trans[:, np.newaxis], axis=1) * dlam
    return fluxes

# ------------------------------------------------------------------
# Reconstruction utilities
# ------------------------------------------------------------------
def train_regression_model(fluxes, coeffs):
    """
    Train a linear regression model mapping photometry to basis coefficients.

    Parameters
    ----------
    fluxes : ndarray, shape (n_samples, K)
        Photometric fluxes.
    coeffs : ndarray, shape (n_samples, M)
        True basis coefficients.

    Returns
    -------
    model : LinearRegression
        Fitted regression model.
    """
    model = LinearRegression(fit_intercept=False)
    model.fit(fluxes, coeffs)
    return model

def predict_coefficients(model, fluxes):
    """
    Predict basis coefficients from photometry.

    Parameters
    ----------
    model : LinearRegression
    fluxes : ndarray, shape (n_samples, K)

    Returns
    -------
    coeffs_pred : ndarray, shape (n_samples, M)
    """
    return model.predict(fluxes)

def reconstruct_spectra(coeffs, wavelengths, params):
    """
    Reconstruct spectra from predicted coefficients.

    Parameters
    ----------
    coeffs : ndarray, shape (n_samples, M)
    wavelengths : ndarray, shape (N,)
    params : list of tuples
        Basis parameters.

    Returns
    -------
    spectra_rec : ndarray, shape (n_samples, N)
    """
    basis = generate_gaussian_basis(wavelengths, params)
    return coeffs @ basis.T

# ------------------------------------------------------------------
# Main demonstration
# ------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)

    # Wavelength grid
    wavelengths = np.linspace(400, 800, 401)   # 1 nm resolution

    # Define Gaussian basis
    n_components = 5
    centres = rng.uniform(420, 780, size=n_components)
    widths  = rng.uniform(5, 15, size=n_components)
    amps    = rng.uniform(1, 3, size=n_components)
    gauss_params = [(c, w, a) for c, w, a in zip(centres, widths, amps)]

    # Generate synthetic spectra
    n_samples = 200
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, wavelengths, gauss_params, rng=rng)

    # Define filters
    filter_definitions = [
        {'name': 'F1', 'centre': 475, 'width': 50},
        {'name': 'F2', 'centre': 575, 'width': 50},
        {'name': 'F3', 'centre': 675, 'width': 50}
    ]

    # Generate photometric fluxes
    fluxes = generate_photometry(spectra, wavelengths, filter_definitions)

    # Train regression model
    model = train_regression_model(fluxes, true_coeffs)

    # Predict coefficients
    coeffs_pred = predict_coefficients(model, fluxes)

    # Reconstruct spectra
    spectra_rec = reconstruct_spectra(coeffs_pred, wavelengths, gauss_params)

    # Evaluate reconstruction error
    mse = np.mean((spectra - spectra_rec)**2)
    print(f"Mean squared reconstruction error: {mse:.4e}")

    # Example of one spectrum
    idx = 0
    print("\nOriginal spectrum sample (first 10 points):")
    print(spectra[idx, :10])
    print("Reconstructed spectrum sample (first 10 points):")
    print(spectra_rec[idx, :10])

if __name__ == "__main__":
    main()