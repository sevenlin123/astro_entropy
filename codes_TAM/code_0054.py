import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Spectral model generation
# ----------------------------------------------------------------------
def blackbody_lambda(lam, T):
    """Planck function in arbitrary units (W sr^-1 m^-2 m^-1)."""
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    lam_m = lam * 1e-9          # nm → m
    numerator = 2.0 * h * c**2 / lam_m**5
    denominator = np.exp(h * c / (lam_m * k * T)) - 1.0
    return numerator / denominator

def generate_synthetic_spectra(n_samples, wavelengths, temperatures):
    """Generate spectra as blackbodies perturbed by random Gaussian noise."""
    spectra = []
    for T in temperatures:
        spec = blackbody_lambda(wavelengths, T)
        noise = np.random.normal(scale=spec.mean() * 0.05, size=spec.shape)
        spectra.append(spec + noise)
    spectra = np.array(spectra)                     # shape (n_samples, n_wavelengths)
    return spectra

# ----------------------------------------------------------------------
# Filter definition and photometry calculation
# ----------------------------------------------------------------------
def gaussian_filter(wavelengths, center, width):
    """Top-hat filter approximated by Gaussian."""
    return np.exp(-0.5 * ((wavelengths - center) / width)**2)

def generate_filters(wavelengths, n_filters=5):
    centers = np.linspace(wavelengths.min()+100, wavelengths.max()-100, n_filters)
    widths = np.full_like(centers, 50.0)  # 50 nm width
    filters = [gaussian_filter(wavelengths, c, w) for c, w in zip(centers, widths)]
    return np.array(filters)             # shape (n_filters, n_wavelengths)

def compute_photometry(spectra, filters):
    """Integrate spectra times filter transmissions."""
    phot = np.trapz(spectra[:, :, None] * filters[None, :, :], axis=2, dx=(spectra.shape[1]-1)/len(spectra[0]))
    return phot                                 # shape (n_samples, n_filters)

# ----------------------------------------------------------------------
# Reconstruction framework
# ----------------------------------------------------------------------
def train_reconstruction_model(train_spec, train_phot, n_components=10):
    pca = PCA(n_components=n_components, svd_solver='full')
    coeffs = pca.fit_transform(train_spec)     # PCA coefficients
    reg = LinearRegression()
    reg.fit(train_phot, coeffs)                # Fit photometry → coefficients
    return pca, reg

def reconstruct_spectrum(phot, pca, reg):
    coeffs_pred = reg.predict(phot.reshape(1, -1))
    spec_recon = pca.inverse_transform(coeffs_pred)
    return spec_recon[0]

# ----------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Define wavelength grid (in nm)
    wavelengths = np.linspace(500, 2500, 200)   # 500–2500 nm, 200 points

    # Generate temperatures for synthetic stars
    temps = np.linspace(3000, 10000, 30)  # 30 stars

    # Generate spectra
    spectra = generate_synthetic_spectra(len(temps), wavelengths, temps)

    # Define photometric filters
    filters = generate_filters(wavelengths, n_filters=5)

    # Compute photometry
    photometry = compute_photometry(spectra, filters)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        photometry, spectra, test_size=0.2, random_state=0
    )

    # Train reconstruction model
    pca, reg = train_reconstruction_model(y_train, X_train, n_components=20)

    # Reconstruct test spectra
    reconstructed = np.array([reconstruct_spectrum(x, pca, reg) for x in X_test])

    # Evaluate reconstruction error (RMSE)
    rmse = np.sqrt(((reconstructed - y_test) ** 2).mean())
    print(f"Reconstruction RMSE: {rmse:.3f}")