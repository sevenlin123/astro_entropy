import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def wavelength_grid(start=400, stop=800, n_points=100):
    return np.linspace(start, stop, n_points)

def gaussian_filter(center, width, wavelengths):
    return np.exp(-0.5 * ((wavelengths - center) / width)**2)

def generate_filters(wavelengths, centers=[450, 550, 650], width=30):
    return np.array([gaussian_filter(c, width, wavelengths) for c in centers])

def generate_synthetic_spectra(n_samples, wavelengths, n_peaks=3):
    spectra = []
    for _ in range(n_samples):
        spec = np.zeros_like(wavelengths)
        for _ in range(n_peaks):
            center = np.random.uniform(wavelengths.min(), wavelengths.max())
            width = np.random.uniform(5, 20)
            amplitude = np.random.uniform(0.5, 1.5)
            spec += amplitude * gaussian_filter(center, width, wavelengths)
        spectra.append(spec)
    return np.vstack(spectra)

def compute_photometry(spectra, filters):
    # integrate: dot product of spectra with each filter (assuming equal spacing)
    return spectra @ filters.T

def train_reconstruction_model(photometry, spectra, alpha=1.0):
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(photometry, spectra)
    return model

def reconstruct_spectrum(model, photometry):
    return model.predict(photometry)

# Main routine
np.random.seed(42)
wavelengths = wavelength_grid()
filters = generate_filters(wavelengths)
n_train, n_test = 200, 50
train_spectra = generate_synthetic_spectra(n_train, wavelengths)
test_spectra = generate_synthetic_spectra(n_test, wavelengths)

train_photometry = compute_photometry(train_spectra, filters)
test_photometry = compute_photometry(test_spectra, filters)

model = train_reconstruction_model(train_photometry, train_spectra)
reconstructed_test = reconstruct_spectrum(model, test_photometry)

mse = mean_squared_error(test_spectra, reconstructed_test)
print(f"Reconstruction MSE on test set: {mse:.4f}")