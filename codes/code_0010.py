
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
Okay, I need to write a Python script that implements an algorithm for spectral reconstruction. The user has given specific requirements, so I have to make sure I address each one carefully.

First, the script needs to define a spectral model. I'll think about using a Gaussian as a simple model because it's easy to implement and works well for demonstration. So, I'll create a function called generate_spectra that produces synthetic spectra using Gaussian peaks.

Next, I need to generate photometric data from these spectra. Photometry typically involves integrating the spectrum over certain bands. For simplicity, I can create a set of passbands and compute the flux in each band for each spectrum. I'll write a function called generate_photometric_data that takes the spectra and the passbands to calculate the photometric measurements.

Then, the main task is to reconstruct the spectrum from the photometric data. This sounds like a regression problem where the photometric data (features) predict the spectral flux (target). I'll use a machine learning model from scikit-learn, like Ridge Regression, which is suitable for this kind of inverse problem. I'll train the model on the generated data and then use it to predict the spectrum from the photometric measurements.

I also need to include plotting. I'll use matplotlib to visualize both the original and reconstructed spectra, as well as the training and testing data points for the model. This will help in visually assessing the performance of the reconstruction.

Another important point is to structure the code into clear functions. I'll define separate functions for generating spectra, photometric data, and the reconstruction model. This makes the code modular and easier to understand.

I must remember to import the necessary libraries: numpy for array operations, scipy for any additional functionalities, and scikit-learn for the regression model. Also, I'll need to set up the plot parameters for consistent plotting.

Lastly, I'll ensure that the script is self-contained. I'll generate synthetic data within the script, train the model, make predictions, and then plot everything. This way, the code can run independently without external inputs.

Putting it all together, I'll outline the steps: define the spectral model, generate synthetic spectra and photometric data, train the regression model, reconstruct the spectrum, and plot the results. Each part will be encapsulated in functions to maintain clarity and reusability.
</think>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

def generate_spectra(n_spectra, wavelengths, n_peaks):
    """Generate synthetic spectral data."""
    spectra = []
    for _ in range(n_spectra):
        # Create a Gaussian-shaped spectrum
        peak_pos = np.random.uniform(400, 700)
        sigma = 50
        flux = np.exp(-((wavelengths - peak_pos)**2)/(2*sigma**2))
        spectra.append(flux)
    return np.array(spectra)

def generate_photometric_bands(wavelengths, n_bands=5):
    """Generate synthetic photometric passband filters."""
    bands = []
    for i in range(n_bands):
        # Randomly positioned Gaussian passband
        center = np.random.uniform(380, 720)
        width = 100
        band = np.exp(-((wavelengths - center)**2)/(2*(width/2)**2))
        bands.append(band)
    return np.array(bands)

def model_reconstruction(photometry, wavelengths, n_components=10):
    """Reconstruct spectrum from photometric data using regression."""
    model = Ridge(alpha=0.1)
    model.fit(photometry, wavelengths)
    reconstructed = model.predict(photometry)
    return reconstructed

# Set up wavelength range
wavelengths = np.linspace(300, 900, 1000)

# Generate synthetic spectra
n_spectra = 10
spectra = generate_spectra(n_spectra, wavelengths, n_peaks=3)

# Generate photometric passbands
passbands = generate_photometric_bands(wavelengths)

# Calculate photometric measurements
photometry = np.dot(passbands, spectra.T).T

# Reconstruct spectra from photometric data
reconstructed = model_reconstruction(photometry, wavelengths)

# Plotting setup
plt.style.use('seaborn')

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot original vs reconstructed spectrum
axes[0].plot(wavelengths, spectra[0], label='Original Spectrum')
axes[0].plot(wavelengths, reconstructed[0], label='Reconstructed')
axes[0].set_xlabel('Wavelength (nm)')
axes[0].set_ylabel('Flux')
axes[0].legend()

# Plot training and testing data relation
axes[1].scatter(photometry[:, 0], wavelengths, c='blue', label='Training Data')
axes[1].scatter(photometry[:, -1], wavelengths, c='red', label='Testing Data')
axes[1].set_xlabel('Photometric Measurement')
axes[1].set_ylabel('Wavelength')
axes[1].legend()

plt.tight_layout()
plt.show()
```