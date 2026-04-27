
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
Okay, I need to figure out how to write this Python code based on the user's requirements. Let me break down what they're asking for.

First, the user wants a minimal Python implementation of an algorithm. The main points are:

1. Define a spectral model.
2. Generate synthetic spectra.
3. Generate photometric data from these spectra.
4. Reconstruct a spectrum from photometric data.

They also specified that the code should use numpy, scipy, and scikit-learn. Plus, it needs plotting and generating synthetic data within the script. Functions should be clearly defined, but no validation methods like LOOCV or cross-validation should be included.

So, I'll start by outlining the steps. I'll need to create a spectral model, which could be something simple like a set of basis vectors. Then, I'll generate synthetic spectra by combining these bases with some coefficients and adding noise. Next, from these spectra, I'll compute photometric data, perhaps by integrating over certain bands or using a set of filters.

For reconstruction, I'll set up an optimization problem where I try to find the coefficients that best reproduce the photometric data. Using least squares seems appropriate here. Once I have the coefficients, I can reconstruct the spectrum.

I'll structure the code into functions: one to generate synthetic spectra, another to compute photometric data, and another for the reconstruction. I'll include plotting to visualize the original and reconstructed spectra, as well as the photometric data.

I'll make sure to import numpy, scipy, and sklearn for any necessary linear algebra operations. Since the code needs to be runnable, I'll include all necessary imports at the beginning.

Putting it all together, I'll define the spectral model in the first function, generate the synthetic data, compute the photometric measurements, perform the reconstruction using least squares, and then plot everything to show the results.

I think that covers all the requirements. Now, I'll translate this plan into code, ensuring each part is clear and concise.
</think>

```python
import numpy as np
from scipy.optimize import least_squares
from matplotlib import pyplot as plt

def generate_synthetic_spectra(n_pixels=100, n_spectra=50):
    """Generate synthetic spectral data."""
    x = np.linspace(0, 10, n_pixels)  # Wavelength grid
    specs = []
    for _ in range(n_spectra):
        coeff = np.random.rand(4)  # Model coefficients
        spec = coeff[0] * np.exp(-x**2 / coeff[1]) + \
               coeff[2] * np.exp(-x**2 / coeff[3])
        specs.append(spec + np.random.normal(0, 0.1, len(x)))
    return np.array(specs)

def compute_photometry(spectra, filter_curves, wavelengths):
    """Compute photometric measurements from spectra."""
    photometry = []
    for spec in spectra:
        flux = np.sum(spec * filter_curves[:, 1]) / np.sum(filter_curves[:, 1])
        photometry.append(flux)
    return np.array(photometry)

def spectral_reconstruction(phot, filter_curves, wavelengths):
    """Reconstruct spectrum from photometric data."""
    A = filter_curves[:, 1]
    b = phot
    coeff, _, rank, singular = np.linalg.lstsq(A, b, rcond=None)
    reconstructed = A.dot(coeff)
    return reconstructed

# Generate synthetic data
np.random.seed(42)
n_pixels = 100
n_spectra = 50

# Define filter curves
filters = np.array([
    [0.3, 1.0], [0.4, 1.0], [0.5, 1.0],  # Simplified filter responses
    [0.6, 1.0], [0.7, 1.0]
])

specs = generate_synthetic_spectra(n_pixels, n_spectra)
phot = compute_photometry(specs, filters, np.linspace(0, 10, n_pixels))

# Reconstruct spectrum
reconstructed_spec = spectral_reconstruction(phot, filters, np.linspace(0, 10, n_pixels))

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(np.linspace(0, 10, 100), specs[0], label='Original Spectrum')
plt.plot(np.linspace(0, 10, 100), reconstructed_spec, label='Reconstructed Spectrum')
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Spectral Reconstruction')
plt.legend()
plt.show()
```