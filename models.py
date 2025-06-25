#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 17:10:11 2025

@author: ymai0110
"""

import sys
sys.path.insert(0, '/Users/ymai0110/Documents/Blobby3D/Krumholz_code/kbfc17/')
from sigma_sf import sigma_sf
import astropy.constants as const
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt


# SFR to Ha conversion factor, Kennicutt & Evans (2012)
Ha_per_SFR = 10.**41.27*(u.erg/u.s)/(u.Msun/u.yr)

# Thermal + turbulent correction for Ha velocity dispersions 
sigma_Ha = 15.0*u.km/u.s

# Fiducial parameter values that we won't alter
sigma_wnm = 5.4
sigma_mol = 0.2
phimp = 1.4
epsff = 0.015
tsfmax = 2.0*u.Gyr
Qmin = 1.0
eta = 1.5 # a scaling factor for the turbulent dissipationrate
phiQ = 2.0 # defined as one plus the ratio of the gas to stellar Q
pmstar = 3000.*u.km/u.s

# Four sets of parameters: dwarf-like, spiral-like, intermediate, hi-z
fsf = np.array([0.2, 0.5, 0.8, 1.0])
vphi = np.array([100., 220., 200, 200.])*u.km/u.s
fgP = np.array([0.9, 0.5, 0.6, 0.7])
fgQ = np.array([0.9, 0.5, 0.6, 0.7])
torb = np.array([100., 200., 200., 200.])*u.Myr
beta = np.array([0.5, 0.0, 0, 0.0])
phia = np.array([1., 1., 2., 3.])
sfr_cut = np.array([0.5, 5., 20., -1.])*u.Msun/u.yr
th_names = ['Local dwarf', 'Local spiral', 'intermediate-$z$', 'High-$z$']


# intermediate-z case
i = 2
fsf_ = fsf[i]
beta_ = beta[i]
torb_ = torb[i]
fgQ_ = fgQ[i]
fgP_ = fgP[i]



############ SF + transport model, eq 59 in M. Krumholz+2018 #########


sigma_th = sigma_wnm*(1.0 - fsf_) + sigma_mol*fsf_

# the velocity dispersion sigma_sf that can be supported by star formation alone
ssf = sigma_sf(sigma_th = sigma_th,
               fsf = fsf_, beta=beta_,
               torb=torb_.to(u.Myr))*u.km/u.s

# Q from 10^4 to 1
Qvec = np.logspace(4, 0, 10000)*Qmin

sigma_vec = np.linspace(1.0, 100.0, 10000)

# const velocity dispersion for SF support only
sigma1 = np.ones(Qvec.shape)*ssf

# calculate the range of SFR surfance density for SF support only
# note it's Sigma_SFR, not SFR
Sigma_SFR1 = fsf_ * np.sqrt(8*(1+beta_))*fgQ_*ssf/(const.G*Qvec*torb_**2)* \
                    np.maximum(
                        np.sqrt(2*(1+beta_)/(3*fgP_*phimp))*8*epsff*fgQ_/Qvec,
                        torb_/tsfmax)

# combine SF feedback and transport support
sigma2 = sigma_vec*ssf
Sigma_SFR2 = fsf_ * np.sqrt(8*(1+beta_))*fgQ_*sigma2/(const.G*Qmin*torb_**2)* \
                    np.maximum(
                        np.sqrt(2*(1+beta_)/(3*fgP_*phimp))*8*epsff*fgQ_/Qmin,
                        torb_/tsfmax)

sigma = np.concatenate((np.array(sigma1.to(u.km/u.s)),
                            np.array(sigma2.to(u.km/u.s)))) * \
                            u.km/u.s
                            
sigma_ion_ft = np.sqrt(sigma ** 2 + sigma_Ha **2)
                            
Sigma_SFR_ft = np.concatenate((np.array(Sigma_SFR1.to(u.Msun/u.yr/u.kpc**2)),
                            np.array(Sigma_SFR2.to(u.Msun/u.yr/u.kpc**2)))) * \
                            u.Msun/u.yr/u.kpc**2
                

plt.scatter(np.log10(Sigma_SFR_ft.value),np.log10(sigma_ion_ft.value))
plt.xlabel('log SigmaSFR [Msun/yr/kpc**2]')
plt.ylabel('log sigma_ion_gas [km/s]]')
plt.title('feedback + transport')
plt.show()

######### feedback only model, eq 61 in M. Krumholz+2018 #######

phint = 1.0 # the fraction of the velocity dispersion that isnonthermal
sigma = sigma_vec* 3 *u.km/u.s # from 3 to 300
p_on_m = 3000 *u.km/u.s

Sigma_SFR_f_k18 = 8 * (beta_ + 1) * np.pi * eta * np.sqrt(phimp*phint**3) * phiQ * \
                    sigma**2 / (const.G*Qmin**2*p_on_m*fgP_*torb_**2)
sigma_ion_f_k18 = np.sqrt(sigma ** 2 + sigma_Ha **2)

plt.scatter(np.log10(Sigma_SFR_f_k18.to(u.Msun/u.yr/u.kpc**2).value),
            np.log10(sigma_ion_f_k18.to(u.km/u.s).value))
plt.xlabel('log SigmaSFR [Msun/yr/kpc**2]')
plt.ylabel('log sigma_ion_gas [km/s]]')
plt.title('feedback only Krumholz+18')
plt.show()

########## feedack only model, TIGRESS, Ostriker+2022, ref. Lenkic+2024 eq27 #

# log sigma_mol = 0.1804 * log SigmaSFR + 1.56

log_SigmaSFR_f_O22 = np.linspace(-3,1.5,1000)
log_sigma_mol = 0.1804 * log_SigmaSFR_f_O22 + 1.56
sigma_ion_f_O22 = np.sqrt(np.power(10,log_sigma_mol)**2 + 15**2)
plt.scatter(log_SigmaSFR_f_O22,np.log10(sigma_ion_f_O22))
plt.xlabel('log SigmaSFR [Msun/yr/kpc**2]')
plt.ylabel('log sigma_ion_gas [km/s]]')
plt.title('feedback only Ostriker+22')
plt.show()

############### compare three models ############
plt.scatter(np.log10(Sigma_SFR_ft.value),np.log10(sigma_ion_ft.value),label='F+T')
plt.scatter(np.log10(Sigma_SFR_f_k18.to(u.Msun/u.yr/u.kpc**2).value),
            np.log10(sigma_ion_f_k18.to(u.km/u.s).value),label='F,K+18')
plt.scatter(log_SigmaSFR_f_O22,np.log10(sigma_ion_f_O22),label='F,O+22')
plt.xlabel('log SigmaSFR [Msun/yr/kpc**2]')
plt.ylabel('log sigma_ion_gas [km/s]]')
plt.legend()
plt.title('compare')
plt.show()
