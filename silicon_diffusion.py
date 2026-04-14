"""
Silicon Dopant Diffusion Simulation
------------------------------------
Models phosphorus diffusion in silicon using an analytical solution
to Fick's second law. Shows how furnace temperature controls
junction depth in semiconductor fabrication.

Author: Nitish Noel Prakash
        MS Materials Science and Engineering, Carnegie Mellon University

Physics
-------
Dopant concentration profile:
    C(x, t) = Cs * erfc( x / (2 * sqrt(D * t)) )

Temperature-dependent diffusivity (Arrhenius):
    D = D0 * exp( -Ea / (kB * T) )

Parameters (phosphorus in silicon):
    D0 = 10.5 cm^2/s    (pre-exponential factor)
    Ea = 3.69 eV        (activation energy)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

os.makedirs("figures", exist_ok=True)

# Physical constants
kB = 8.617e-5       # Boltzmann constant (eV/K)

# Process conditions
Cs    = 1e20        # Surface concentration (cm^-3) -- heavy doping
t     = 30 * 60     # Anneal time: 30 minutes (seconds)
C_bg  = 1e15        # Background doping (cm^-3) -- defines junction

# Phosphorus in silicon diffusion parameters
D0 = 10.5           # cm^2/s
Ea = 3.69           # eV

# Depth array: 0 to 5 microns
x_um = np.linspace(0, 5, 2000)
x_cm = x_um * 1e-4

# Temperatures to compare
temps_C = [900, 1000, 1100]
colors  = ['#185FA5', '#1D9E75', '#D85A30']

plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
})


# Figure 1: Concentration profiles at each temperature
fig, ax = plt.subplots(figsize=(9, 5.5))

for T_C, color in zip(temps_C, colors):
    T_K = T_C + 273.15
    D   = D0 * np.exp(-Ea / (kB * T_K))            # Arrhenius diffusivity
    C   = Cs * erfc(x_cm / (2 * np.sqrt(D * t)))   # erfc concentration profile
    C   = np.maximum(C, 1e10)                        # floor at physical minimum

    ax.semilogy(x_um, C, color=color, linewidth=2.5, label=f'{T_C} \u00b0C')

ax.axhline(C_bg, color='#aaa', linewidth=1.5, linestyle='--', label='Background doping')

ax.set_xlabel('Depth into wafer (\u03bcm)', fontsize=13)
ax.set_ylabel('Phosphorus concentration (cm\u207b\u00b3)', fontsize=13)
ax.set_title('Where phosphorus ends up after 30 minutes\nHigher temperature = deeper penetration', fontsize=13, pad=14)
ax.set_xlim(0, 5)
ax.set_ylim(1e10, 2e21)
ax.legend(fontsize=11, frameon=False)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('figures/fig1_diffusion_profiles.png', dpi=180, bbox_inches='tight')
plt.close()
print('Saved: figures/fig1_diffusion_profiles.png')


# Figure 2: Junction depth across full temperature range
temp_range = np.linspace(850, 1150, 300)
jd = []

for T_C in temp_range:
    T_K = T_C + 273.15
    D   = D0 * np.exp(-Ea / (kB * T_K))
    C   = Cs * erfc(x_cm / (2 * np.sqrt(D * t)))
    idx = np.where(C <= C_bg)[0]
    jd.append(x_um[idx[0]] if len(idx) > 0 else np.nan)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(temp_range, jd, color='#185FA5', linewidth=2.5)
ax.fill_between(temp_range, jd, alpha=0.1, color='#185FA5')

for T_C, color in zip(temps_C, colors):
    T_K  = T_C + 273.15
    D    = D0 * np.exp(-Ea / (kB * T_K))
    C    = Cs * erfc(x_cm / (2 * np.sqrt(D * t)))
    idx  = np.where(C <= C_bg)[0]
    if len(idx):
        jd_val = round(x_um[idx[0]], 2)
        ax.scatter(T_C, jd_val, color=color, s=80, zorder=5)
        ax.annotate(f'{T_C}\u00b0C  {jd_val}\u03bcm',
                    xy=(T_C, jd_val), xytext=(T_C + 10, jd_val + 0.04),
                    fontsize=10, color=color)

ax.set_xlabel('Furnace temperature (\u00b0C)', fontsize=13)
ax.set_ylabel('Junction depth (\u03bcm)', fontsize=13)
ax.set_title('Junction depth rises nonlinearly with temperature\n30-minute anneal, phosphorus in silicon', fontsize=13, pad=14)
ax.set_xlim(850, 1150)
ax.set_ylim(0, 1.6)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('figures/fig2_junction_depth.png', dpi=180, bbox_inches='tight')
plt.close()
print('Saved: figures/fig2_junction_depth.png')

print('\nDone. Run this script from the repo root. Figures saved to figures/')
