import numpy as np
from sklearn.linear_model import Ridge


def create_gaussian_basis(n_bases: int, wavelengths: np.ndarray) -> np.ndarray:
    """Generate Gaussian basis functions."""
    width = 300.0  # angstroms
    centers = np.linspace(
        wavelengths[0] + 500, wavelengths[-1] - 500, n_bases
    )
    basis = np.array(
        [
            np.exp(-((wavelengths - c) ** 2) / (2 * width ** 2))
            for c in centers
        ]
    )
    return basis  # shape (n_bases, n_wavelengths)


def generate_coeffs(n_samples: int, n_bases: int) -> np.ndarray:
    """Randomly sample coefficients for each basis function."""
    return np.random.uniform(-1, 1, size=(n_samples, n_bases))


def generate_spectra(
    coeffs: np.ndarray,
    basis_funcs: np.ndarray,
    noise_std: float = 0.02,
) -> np.ndarray:
    """Build spectra from coefficients and basis functions."""
    spectra = coeffs @ basis_funcs  # (n_samples, n_wavelengths)
    spectra += np.random.normal(scale=noise_std, size=spectra.shape)
    return spectra


def create_filters() -> dict:
    """Define simple rectangular filter bandpasses."""
    return {
        "U": (3500, 4000),
        "B": (4000, 5000),
        "V": (5000, 6000),
        "R": (6000, 7000),
        "I": (7000, 8000),
        "J": (12000, 13000),
        "H": (16000, 17000),
        "K": (20000, 21000),
    }


def compute_photometry(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    filters: dict,
) -> np.ndarray:
    """Compute mean flux in each filter band."""
    n_samples = spectra.shape[0]
    n_filters = len(filters)
    phot = np.empty((n_samples, n_filters))
    for idx, (name, (wave_min, wave_max)) in enumerate(filters.items()):
        mask = (wavelengths >= wave_min) & (wavelengths <= wave_max)
        phot[:, idx] = spectra[:, mask].mean(axis=1)
    return phot  # shape (n_samples, n_filters)


def train_regression(phot: np.ndarray, coeffs: np.ndarray) -> Ridge:
    """Fit a multi‑output ridge regression model."""
    model = Ridge(alpha=1.0, fit_intercept=False)
    model.fit(phot, coeffs)
    return model


def reconstruct_spectrum(
    photometry: np.ndarray,
    model: Ridge,
    basis_funcs: np.ndarray,
) -> np.ndarray:
    """Predict coefficients from photometry and build the spectrum."""
    coeff_pred = model.predict(photometry)
    return coeff_pred @ basis_funcs


def main():
    rng = np.random.default_rng(seed=42)

    # Setup
    wavelengths = np.linspace(3000, 25000, 400)  # angstroms
    n_bases = 12
    basis_funcs = create_gaussian_basis(n_bases, wavelengths)

    # Generate training data
    n_train = 1500
    coeffs_train = generate_coeffs(n_train, n_bases)
    spectra_train = generate_spectra(coeffs_train, basis_funcs)
    filters = create_filters()
    phot_train = compute_photometry(spectra_train, wavelengths, filters)

    # Train regression model
    model = train_regression(phot_train, coeffs_train)

    # Test on a new synthetic star
    coeff_true = np.array([0.5, -0.3, 0.8, -0.1, 0.4, 0.0, -0.6, 0.7, -0.2, 0.3, 0.1, -0.4])
    spec_true = coeff_true @ basis_funcs
    phot_test = np.array([spec_true[(wavelengths >= w_min) & (wavelengths <= w_max)].mean()
                          for _, (w_min, w_max) in filters.items()]).reshape(1, -1)

    # Reconstruct
    spec_recon = reconstruct_spectrum(phot_test, model, basis_funcs)[0]

    # Compare
    print("True spectrum sample:", spec_true[:10])
    print("Reconstructed spectrum sample:", spec_recon[:10])
    diff = np.abs(spec_true - spec_recon)
    print(f"Mean absolute reconstruction error: {diff.mean():.4f}")


if __name__ == "__main__":
    main()