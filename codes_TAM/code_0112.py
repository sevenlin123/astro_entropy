import numpy as np
from sklearn.linear_model import Ridge

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def build_basis(wavelengths, mus, sigma):
    return np.vstack([gaussian(wavelengths, mu, sigma) for mu in mus]).T

def build_filters(wavelengths, centers, widths):
    return [gaussian(wavelengths, c, w) for c, w in zip(centers, widths)]

def compute_filter_matrix(basis_matrix, filters, wavelengths):
    n_filters = len(filters)
    n_basis = basis_matrix.shape[1]
    M = np.zeros((n_filters, n_basis))
    for j, filt in enumerate(filters):
        for i in range(n_basis):
            M[j, i] = np.trapz(basis_matrix[:, i] * filt, wavelengths)
    return M

def generate_synthetic_spectra(n_samples, wavelengths, basis_mus, basis_sigma, noise_std=0.02):
    basis = build_basis(wavelengths, basis_mus, basis_sigma)
    weights = np.random.uniform(-1, 1, size=(n_samples, basis.shape[1]))
    spectra = weights @ basis.T
    spectra += np.random.normal(scale=noise_std, size=spectra.shape)
    return weights, spectra

def generate_photometric_data(spectra, filters, wavelengths):
    photometry = []
    for spec in spectra:
        fluxes = [
            np.trapz(spec * filt, wavelengths) / np.trapz(filt, wavelengths)
            for filt in filters
        ]
        photometry.append(fluxes)
    return np.array(photometry)

def reconstruct_spectra_from_photometry(photometry, filter_matrix, basis_matrix, alpha=1.0):
    n_samples = photometry.shape[0]
    n_basis = basis_matrix.shape[1]
    reconstructed_weights = np.zeros((n_samples, n_basis))
    for i in range(n_samples):
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(filter_matrix, photometry[i])
        reconstructed_weights[i] = ridge.coef_
    reconstructed_spectra = reconstructed_weights @ basis_matrix.T
    return reconstructed_weights, reconstructed_spectra

def main():
    wavelengths = np.linspace(300, 800, 250)
    basis_mus = [350, 400, 450, 500, 550, 600]
    basis_sigma = 20
    n_samples = 10

    basis_matrix = build_basis(wavelengths, basis_mus, basis_sigma)
    filter_centers = [360, 440, 530]
    filter_widths = [30, 30, 30]
    filters = build_filters(wavelengths, filter_centers, filter_widths)
    filter_matrix = compute_filter_matrix(basis_matrix, filters, wavelengths)

    true_weights, spectra = generate_synthetic_spectra(
        n_samples, wavelengths, basis_mus, basis_sigma
    )
    photometry = generate_photometric_data(spectra, filters, wavelengths)

    rec_weights, rec_spectra = reconstruct_spectra_from_photometry(
        photometry, filter_matrix, basis_matrix
    )

    mse = np.mean((spectra - rec_spectra) ** 2)
    print(f"Mean squared error between true and reconstructed spectra: {mse:.4f}")

if __name__ == "__main__":
    main()