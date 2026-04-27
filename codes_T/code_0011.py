import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# Wavelength grid (nm)
wavelengths = np.linspace(300, 900, 601)

# Basis functions
def continuum(x):
    return np.ones_like(x)

def linear(x):
    return x - np.mean(x)

def gaussian(x, mu=500, sigma=50):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Spectral model
def spectral_model(params, x=wavelengths):
    return params[0] * continuum(x) + params[1] * linear(x) + params[2] * gaussian(x)

# Filter transmissions
filters = []
for center in [400, 600, 800]:
    filt = np.exp(-0.5 * ((wavelengths - center) / 30) ** 2)
    filters.append(filt)
filters = np.array(filters)

# Photometry from spectra
def photometry_from_spectrum(flux):
    phots = []
    for filt in filters:
        phots.append(simps(flux * filt, wavelengths) / simps(filt, wavelengths))
    return np.array(phots)

# Generate synthetic dataset
def generate_dataset(n):
    params = np.random.uniform(-1, 1, size=(n, 3))
    spectra = np.array([spectral_model(p) for p in params])
    phots = np.array([photometry_from_spectrum(f) for f in spectra])
    return params, phots, spectra

# Reconstruction pipeline
class SpectrumReconstructor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, phots, params):
        self.model.fit(phots, params)

    def predict(self, phots):
        return self.model.predict(phots)

    def reconstruct(self, params):
        return spectral_model(params)

# Main demonstration
if __name__ == "__main__":
    # Training data
    train_params, train_phots, _ = generate_dataset(200)
    recon = SpectrumReconstructor()
    recon.fit(train_phots, train_params)

    # Test on new data
    test_params, test_phots, true_spectra = generate_dataset(10)
    pred_params = recon.predict(test_phots)
    reconstructed = np.array([recon.reconstruct(p) for p in pred_params])

    # Show first test case
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(wavelengths, true_spectra[0], label='True')
    plt.plot(wavelengths, reconstructed[0], '--', label='Reconstructed')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.legend()
    plt.show()