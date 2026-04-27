
Write a minimal Python implementation of the following algorithm.

Requirements:

- implement only the core reconstruction framework
- ignore validation methods (LOOCV, cross-validation)
- need plotting
- generate synthetic example input data inside the script
- Do NOT show reasoning or thinking.
- Do NOT explain the code.
- Return only the Python code.

The code should:

1. define a spectral model
2. generate synthetic spectra
2. generate photometric data from synthetic spectra
3. reconstruct a synthetic spectrum from photometruic

Constraints:

- use numpy, scipy, and scikit-learn
- produce runnable Python code
- define clear functions

Return only the code.
</think>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def generate_synthetic_spectra(n_samples, n_features, noise=0.1):
    """Generate synthetic spectral data."""
    X = np.random.rand(n_samples, n_features) * 2 - 1
    return X + np.random.normal(0, noise, size=X.shape)

def generate_photometric_data(spectra, response_curves):
    """Simulate photometric measurements from spectra using response curves."""
    return np.dot(spectra, response_curves)

def spectral_reconstructor(response_curves):
    """Reconstruct spectral features from photometric measurements."""
    pca = PCA()
    scaler = StandardScaler()
    return make_pipeline(scaler, pca)

def main():
    """Main function to demonstrate spectral reconstruction."""
    n_samples = 100
    n_features = 1000
    n_filters = 50

    # Generate synthetic spectra
    spectra = generate_synthetic_spectra(n_samples, n_features)

    # Define response curves for each filter
    response_curves = np.random.rand(n_filters, n_features)

    # Generate photometric observations
    photometric = generate_photometric_data(spectra, response_curves)

    # Train the spectral reconstructor
    model = spectral_reconstructor(response_curves)
    reconstructed_spectra = model.fit_transform(photometric)

    # Plot original vs. reconstructed spectra
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(spectra[:5], cmap='gray', aspect='auto')
    plt.title('Original Spectra')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_spectra[:5], cmap='gray', aspect='auto')
    plt.title('Reconstructed Spectra')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
```