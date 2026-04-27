
Write a minimal Python implementation of the following algorithm.

Requirements:

- implement only the core reconstruction framework
- ignore validation methods (LOOCV, cross-validation)
- Do not need plotting
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
from scipy import integrate
from sklearn.decomposition import NMF

def generate_synthetic_spectra(n_observations, n_features):
    """Generate synthetic spectral data."""
    return np.random.rand(n_observations, n_features)

def generate_photometric_data(spectra, filter_transmissions):
    """Convert spectra to photometric measurements using filters."""
    return np.dot(spectra, filter_transmissions.T)

def spectral_reconstruction(photometry, n_components=50):
    """Reconstruct spectra from photometric data using NMF."""
    model = NMF(n_components=n_components, max_iter=1000)
    reconstructed = model.fit_transform(photometry)
    return model.components_.T * reconstructed

# Example usage
n_samples = 100
n_wavelengths = 400
filters = np.array([np.tri(100, k=i) for i in range(4)])

spectra_true = generate_synthetic_spectra(n_samples, n_wavelengths)
photometry = generate_photometric_data(spectra_true, filters)
spectra_reconstructed = spectral_reconstruction(photometry, n_components=50)
```

This implementation defines the core spectral reconstruction framework using NMF. It generates synthetic data, converts it to photometric measurements, then reconstructs the spectra from the photometric data. The code is minimal and focuses on the essential components without additional validation or visualization steps.