
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
Okay, I need to write a Python code that implements a spectral unmixing algorithm based on the given steps. Let me break it down.

First, I should define a spectral model. The model will have a set number of components, say n_components. Using sklearn's NMF makes sense here because it's a common method for spectral analysis.

Next, I need to generate synthetic spectra. I'll create a base matrix with a few pure spectra. Then, using these, I'll mix them with random weights to form the observed data. Adding some noise will make it more realistic.

Then, I'll generate photometric data from these spectra. Since photometric bands are at specific wavelengths, I'll extract the spectra values at those wavelengths to form the photometric data matrix.

Now, for reconstruction. I'll apply the NMF model to the photometric data to estimate the mixing coefficients. Once I have these coefficients, I'll multiply them by the basis vectors (the original pure spectra) to reconstruct the synthetic spectrum.

I should structure this into functions: one for generating synthetic spectra, another for creating photometric data, and another for the reconstruction process. Each function should be clear and concise.

I also need to ensure that the code is runnable, so I'll include necessary imports like numpy, scipy, and sklearn. I'll avoid any plotting or validation methods as per the requirements.

Let me put it all together step by step. Define the model, generate the synthetic data, extract photometric bands, apply NMF, and reconstruct the spectrum. Make sure everything is wrapped in functions and runs smoothly.
</think>

```python
import numpy as np
from sklearn.decomposition import NMF

def define_spectral_model(n_components):
    model = NMF(n_components=n_components, random_state=42)
    return model

def generate_synthetic_spectra(n_samples, n_features, n_components):
    # Create synthetic spectra with n_components pure components
    rng = np.random.RandomState(42)
    concentrations = rng.uniform(0, 1, n_samples)
    components = np.exp(rng.normal(0, 0.5, (n_components, n_features)))
    synthetic = np.zeros((n_samples, n_features))
    for i in range(n_components):
        synthetic += concentrations[i] * components[i]
    return synthetic, concentrations, components

def generate_photometric_data(spectra, band_indices):
    photometric = np.zeros((len(band_indices), spectra.shape[1]))
    for i, idx in enumerate(band_indices):
        photometric[i] = spectra[:, idx]
    return photometric

def reconstruct_spectrum(photometric, basis_vectors):
    nmf = NMF(random_state=42).fit_transform(photometric, basis_vectors)
    reconstructed = np.dot(nmf, basis_vectors)
    return reconstructed

# Example usage
np.random.seed(42)
n_samples, n_features = 10, 200
synthetic_spectra, _, basis = generate_synthetic_spectra(n_samples, n_features, 3)
band_indices = [50, 100, 150]  # Example photometric bands
photometric = generate_photometric_data(synthetic_spectra, band_indices)
reconstructed = reconstruct_spectrum(photometric, basis)

print("Synthetic Spectra Shape:", synthetic_spectra.shape)
print("Photometric Data Shape:", photometric.shape)
print("Reconstructed Spectrum Shape:", reconstructed.shape)
```