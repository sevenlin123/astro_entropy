
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
Okay, I need to write a minimal Python implementation of an algorithm that does spectral reconstruction using photometric data. The user has specified certain requirements, so I should carefully follow them.

First, I need to understand the steps involved. The algorithm should:

1. Define a spectral model. So, I'll create a function that generates a smooth spectrum based on a set of parameters. This function will take parameters like temperature and metallicity, but for simplicity, maybe I can just use a blackbody curve as the spectral model.

2. Generate synthetic spectra. I'll simulate some data by applying noise and instrumental response to the generated spectra. This means creating a function that takes the model and adds Gaussian noise and a response curve.

3. Generate photometric data from these spectra. Photometry involves integrating the spectrum over passbands. I'll define a set of passbands (like u, g, r, etc.) and compute the flux in each band from the synthetic spectra.

4. Reconstruct a synthetic spectrum from photometric data. This step involves inverting the problem: given the photometric magnitudes, recover the original spectrum. I'll set up an optimization problem where I minimize the difference between the model spectrum and the observed photometry.

Now, considering the constraints: I must use numpy, scipy, and scikit-learn. The code should be runnable, define clear functions, and include synthetic data generation within the script.

I'll structure the code with functions. First, define the spectral model as a blackbody. Then, create a function to generate synthetic spectra with noise and response. Next, a function to compute photometric data from the spectra. Finally, set up and solve the optimization problem to reconstruct the spectrum from the photometry.

I need to ensure that all necessary imports are included, like numpy, scipy.optimize, and sklearn.preprocessing for scaling the parameters.

Wait, the spectral model function is called get_spectrum, which returns a wavelength array and flux array. Then, add_noise_and_response applies a Gaussian noise and a response function. The response could be a simple boxcar for simplicity.

For photometric data, I have predefined passbands. The compute_photometry function will integrate the spectrum over each passband. Each passband is represented by its transmission curve and wavelength range. For simplicity, I can assume the passbands are contiguous and non-overlapping.

In the main part, I generate the true parameters, generate the synthetic spectrum, compute the photometry, then scale the parameters and set up an optimization problem using scipy's least squares method. The objective function will compute the difference between the model's photometry and the observed values.

I should also include a note about the limitations, like assuming a single parameter model and simplistic passbands. But since the user only wants the code, I won't include explanations, just the code.

Now, putting it all together, the code defines the necessary functions, generates synthetic data, computes photometry, and reconstructs the spectrum. It uses the required libraries and follows the constraints.
</think>

```python
import numpy as np
from scipy.optimize import leastsq
from sklearn.preprocessing import StandardScaler

def get_spectrum(parameters):
    # Simple blackbody model as a spectral representation
    wavelengths = np.logspace(-1, 1, 100)  # Angstrom
    flux = np.exp(-((wavelengths - parameters[0])**2)/(2*parameters[1]**2))
    return wavelengths, flux

def add_noise_and_response(spectrum, noise_level=0.1):
    # Simulate noise and instrumental response
    noise = np.random.normal(0, noise_level, len(spectrum))
    response = np.convolve(spectrum, np.ones(5), mode='valid') / 5
    noisy_spectrum = spectrum + noise
    return noisy_spectrum

def compute_photometry(spectrum, passbands):
    photometry = []
    for band in passbands:
        trans = band['transmission']
        spec = spectrum[band['wrange'][0]:band['wrange'][1]]
        integrated_flux = np.sum(spec * trans)
        photometry.append(integrated_flux)
    return photometry

def model_photometry(params, wavelengths, passbands):
    # Predict photometry from spectral model parameters
    _, flux_pred = get_spectrum(params)
    phot_pred = compute_photometry(flux_pred, passbands)
    return phot_pred

def residual(params, wavelengths, phot_obs, passbands):
    phot_mod = model_photometry(params, wavelengths, passbands)
    return phot_obs - phot_mod

# Synthetic data setup
true_params = [0.5, 0.3]  # Simplified spectral parameters
wavelengths_true, flux_true = get_spectrum(true_params)

# Add noise and response
noisy_flux = add_noise_and_response(flux_true)

# Define simple passbands (example)
passbands = [
    {'name': 'u', 'transmission': np.ones(10), 'wrange': (0, 10)},
    {'name': 'g', 'transmission': np.zeros(10), 'wrange': (10, 20)},
    {'name': 'r', 'transmission': np.ones(10), 'wrange': (20, 30)}
]

# Compute photometric measurements
phot_obs = compute_photometry(noisy_flux, passbands)

# Scale features for optimization
scaler = StandardScaler()
phot_obs_scaled = scaler.fit_transform(phot_obs.reshape(-1, 1))

# Set up and solve the optimization problem
initial_guess = [0.6, 0.4]
result = leastsq(residual, initial_guess, args=(wavelengths_true, phot_obs_scaled, passbands))

# Reconstructed parameters
reconstructed_params = result.x

print("True parameters:", true_params)
print("Reconstructed parameters:", reconstructed_params)
```