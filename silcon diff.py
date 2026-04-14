import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Inputs
Cs = 1e20          # surface concentration, cm^-3
D = 1e-13          # diffusivity, cm^2/s
t = 30 * 60        # time in seconds (30 min)

# Depth range: 0 to 5 micrometers
x_um = np.linspace(0, 5, 500)
x_cm = x_um * 1e-4   # convert micrometers to cm

# Constant-source diffusion equation
C = Cs * erfc(x_cm / (2 * np.sqrt(D * t)))

# Plot
plt.figure(figsize=(8, 5))
plt.semilogy(x_um, C, linewidth=2)
plt.xlabel("Depth (µm)")
plt.ylabel("Dopant Concentration (cm$^{-3}$)")
plt.title("Constant-Source Dopant Diffusion in Silicon")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

