
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
from scipy import integrate
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

def spectral_model(wavelengths, coefficients):
    return np.sum(coefficients[:, i] * basis_functions[i](wavelengths) for i in range(len(basis_functions)))

basis_functions = [
    lambda x: np.sin(x),
    lambda x: np.cos(x),
    lambda x: x,
    lambda x: 1.0
]

# Generate synthetic spectra
n_pixels = 100
wavelength_range = np.linspace(400, 700, n_pixels)
true_coefficients = np.array([[0.5, 1.0, -0.3, 0.8]])

synthetic_spectra = spectral_model(wavelength_range, true_coefficients)

# Generate photometric data
bandpasses = ['u', 'g', 'r', 'i', 'z']
fluxes = []
for bp in bandpasses:
    filter_response = np.loadtxt(f'filter_{bp}.txt')
    convolved = np.convolve(synthetic_spectra, filter_response, mode='same')
    flux = integrate.trapz(convolved, wavelength_range)
    fluxes.append(flux)

# Reconstruct synthetic spectrum from photometric data
nmf = NMF(n_components=4, random_state=42)
reconstructed_coefficients = nmf.fit_transform(np.array(fluxes).reshape(1, -1))

reconstructed_spectra = spectral_model(wavelength_range, reconstructed_coefficients)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(wavelength_range, synthetic_spectra, label='True Spectrum')
plt.plot(wavelength_range, reconstructed_spectra, label='Reconstructed Spectrum')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Flux')
plt.legend()
plt.show()
```