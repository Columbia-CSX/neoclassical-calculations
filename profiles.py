"""
This script generates (using a helper script from writeInput.py)
a BEAMS3D style input file for the stelloptPlusSfincs code to use
to make a sfincs style "profiles" file. Profiles are specified
by some hardcoded arrays which are upsampled using splines (upsampling
is done with the intent to improve the accuracy of the derivative
computation done in stelloptPlusSfincs, since what sfincs really cares
about is the gradient profiles)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
from writeInput import *

# hardcoded arrays from staring at Ken Hammond's thesis for a long time
rho  = np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.40, 0.50, 0.60, 0.70, 0.80, 0.9, 1.00])
n_72 = np.array([0.7, 0.7, 0.72, 0.8, 0.9, 0.90, 0.87, 0.85, 0.83, 0.78, 0.5, 0.43]) # 10^{16} m^{-3}
t_72 = np.array([7.5, 7.4, 7.00, 4.7, 4.7, 4.75, 4.83, 4.83, 4.80, 4.70, 5.0, 5.00]) # eV

# adjusted so that there's no dip on the magnetic axis
n_72_adjusted = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.89, 0.87, 0.85, 0.83, 0.78, 0.5, 0.43]) # 10^{16} m^{-3}
t_72_adjusted = np.array([7.5, 7.4, 7.00, 6.9, 6.7, 5.4, 4.83, 4.68, 4.51, 4.45, 4.40, 4.37]) # eV

# new profiles -- linear in temperature
# loosely inspired by https://www.researchgate.net/figure/HSX-stellarator-profiles-of-a-electron-temperature-and-b-density-in-QHS-and-mirror_fig23_228707435

### ----------------------------------------------------------- ###
###   USER PART: SPECIFY ELECTRON TEMPERATURE t AND DENSITY n
### ----------------------------------------------------------- ###
ne = 1e17 * np.exp( (-1.0*rho/1.11)**3 )
Te = 7.5 * (1 - (1/1.2)*rho)

temperature_multiplier = 0.3 # ratio Ti/Te
output_file_name = f"profile.opt"
### ----------------------------------------------------------- ###
###                      END USER PART                          ###
### ----------------------------------------------------------- ###

n_adjusted = ne
t_adjusted = Te
psi  = rho**2

### for upsampling of the hardcoded arrays (should make Brandon's derivative scheme
### more accurate, especially near the core and edge)

lam = 0.0001
n_72_spline = make_smoothing_spline(psi, n_adjusted, lam=lam)
lam = 0.1
t_72_spline = make_smoothing_spline(psi, t_adjusted, lam=lam)
psi_many = np.linspace(0, 1, 100)

### plotting
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(211)
# # ax.plot(rho, n_72, color='black')
# ax.plot(psi_many, n_72_spline(psi_many)*1e-17, color='black')
# ax.set_ylim(0, 1.2)
# ax.set_xlim(0, 1)
# ax.set_ylabel(r"$n_e$ ($10^{17}\text{m}^{-3}$)")
# ax = fig.add_subplot(212)
# ax.set_xlabel(r"$\sqrt{\psi_N}$")
# # ax.set_xlabel(r"$\rho$")
# ax.set_ylabel(r"$T_e$ ($\text{eV}$)")
# plt.plot(psi_many, t_72_spline(psi_many), color='black')
# # plt.plot(rho, t_72, color='black')
# ax.set_xlim(0, 1)
# plt.ylim(0, 8)
# plt.show()
# fig.savefig("profiles_vs_psiN_steep.png")

NE_AUX_S = np.linspace(0, 1, 20)
NE_AUX_F = n_72_spline(NE_AUX_S)
TE_AUX_S = NE_AUX_S
TE_AUX_F = t_72_spline(TE_AUX_S)*temperature_multiplier
NI_AUX_S = NE_AUX_S
NI_AUX_F = np.vstack([NE_AUX_F])
TI_AUX_S = NE_AUX_S
TI_AUX_F = TE_AUX_F*0.3
NI_AUX_M = np.array([4.6518341428*10**(-26)])
NI_AUX_Z = np.array([1], dtype=np.int32)

outfile = output_file_name
write_profiles(
    outfile,
    NE_AUX_S, NE_AUX_F,
    TE_AUX_S, TE_AUX_F,
    NI_AUX_S, NI_AUX_F,
    TI_AUX_S, TI_AUX_F,
    NI_AUX_M, NI_AUX_Z
)
print(f"Wrote {outfile}")
