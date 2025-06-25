#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:57:32 2025

calculate ionised gas velocity dispersion from SigmaSFR for models

@author: ymai0110
"""

import sys
sys.path.insert(0, '/Users/ymai0110/Documents/Blobby3D/Krumholz_code/kbfc17/')
from sigma_sf import sigma_sf
import astropy.constants as const
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

def K18_F_and_T(SigmaSFR):
    '''
    Krumholz+2018, feedback + Transport model
    
    ############ SF + transport model, eq 59 in M. Krumholz+2018 #########

    Parameters
    ----------
    SigmaSFR : str
        SFR surfance density in M_sun yr^-1 kpc^-2

    Returns
    -------
    sigma : str
        ionised gas velocity dispersion in km s^-1

    '''
    

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
    
    #### find the break point of SigmaSFR
    
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
    break_point_SigmaSFR = np.max(Sigma_SFR1.to(u.Msun/u.yr/u.kpc**2).value)
    
    ssf_plus_ha = np.sqrt(ssf**2 + sigma_Ha**2)
    
    if SigmaSFR <= break_point_SigmaSFR:
        return ssf_plus_ha.to(u.km/u.s).value
    
    
    # if the SigmaSFR is greater than break point, calculate the F+T sigma:
        
    
                        
    sigma2 = SigmaSFR*u.Msun/u.yr/u.kpc**2 \
                /(fsf_ * np.sqrt(8*(1+beta_))*fgQ_/(const.G*Qmin*torb_**2)* \
                        np.maximum(
                            np.sqrt(2*(1+beta_)/(3*fgP_*phimp))*8*epsff*fgQ_/Qmin,
                            torb_/tsfmax))
    
    sigma2_plus_ha = np.sqrt(sigma2**2 + sigma_Ha**2)
    return sigma2_plus_ha.to(u.km/u.s).value


def K18_F_and_T_quick(SigmaSFR=None,get_break_point=False,get_Fonly_sigma=False):
    '''
    quick version of K18_F_and_T
    
    
    only provide the array that SigmaSFR > break point
    
    
    Krumholz+2018, feedback + Transport model
    
    ############ SF + transport model, eq 59 in M. Krumholz+2018 #########

    Parameters
    ----------
    SigmaSFR : array
        SFR surfance density in M_sun yr^-1 kpc^-2

    Returns
    -------
    sigma : array
        ionised gas velocity dispersion in km s^-1

    '''
    

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
    
    #### find the break point of SigmaSFR
    
    sigma_th = sigma_wnm*(1.0 - fsf_) + sigma_mol*fsf_

    # the velocity dispersion sigma_sf that can be supported by star formation alone
    ssf = sigma_sf(sigma_th = sigma_th,
                   fsf = fsf_, beta=beta_,
                   torb=torb_.to(u.Myr))*u.km/u.s

    # Q from 10^4 to 1
    Qvec = np.logspace(4, 0, 10000)*Qmin

    sigma_vec = np.linspace(1.0, 100.0, 1000)

    # const velocity dispersion for SF support only
    sigma1 = np.ones(Qvec.shape)*ssf

    # calculate the range of SFR surfance density for SF support only
    # note it's Sigma_SFR, not SFR
    Sigma_SFR1 = fsf_ * np.sqrt(8*(1+beta_))*fgQ_*ssf/(const.G*Qvec*torb_**2)* \
                        np.maximum(
                            np.sqrt(2*(1+beta_)/(3*fgP_*phimp))*8*epsff*fgQ_/Qvec,
                            torb_/tsfmax)
    break_point_SigmaSFR = np.max(Sigma_SFR1.to(u.Msun/u.yr/u.kpc**2).value)
    
    if get_break_point:
        return break_point_SigmaSFR
    
    
    ssf_plus_ha = np.sqrt(ssf**2 + sigma_Ha**2)
    
    if get_Fonly_sigma:
        return ssf_plus_ha.to(u.km/u.s).value
    
    
    
    
    # if the SigmaSFR is greater than break point, calculate the F+T sigma:
        
    
                        
    sigma2 = SigmaSFR*u.Msun/u.yr/u.kpc**2 \
                /(fsf_ * np.sqrt(8*(1+beta_))*fgQ_/(const.G*Qmin*torb_**2)* \
                        np.maximum(
                            np.sqrt(2*(1+beta_)/(3*fgP_*phimp))*8*epsff*fgQ_/Qmin,
                            torb_/tsfmax))
    
    sigma2_plus_ha = np.sqrt(sigma2**2 + sigma_Ha**2)
    return sigma2_plus_ha.to(u.km/u.s).value


def K18_F_only(SigmaSFR):
    '''
    Krumholz+2018, feedback only model
    
    ######### feedback only model, eq 61 in M. Krumholz+2018 #######

    Parameters
    ----------
    SigmaSFR : str
        SFR surfance density in M_sun yr^-1 kpc^-2

    Returns
    -------
    sigma : str
        ionised gas velocity dispersion in km s^-1

    '''
    
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
    
    phint = 1.0 # the fraction of the velocity dispersion that isnonthermal
    
    p_on_m = 3000 *u.km/u.s
    
    
                        
    sigma_sq = SigmaSFR*u.Msun/u.yr/u.kpc**2 * (const.G*Qmin**2*p_on_m*fgP_*torb_**2)/ \
                (8 * (beta_ + 1) * np.pi * eta * np.sqrt(phimp*phint**3) * phiQ)
    sigma = np.sqrt(sigma_sq)
    sigma_plus_ha = np.sqrt(sigma**2 + sigma_Ha**2)
    
    return sigma_plus_ha.to(u.km/u.s).value

def O22_F_only(SigmaSFR):
    '''
    ########## feedack only model, TIGRESS, Ostriker+2022, ref. Lenkic+2024 eq27 #

    # log sigma_mol = 0.1804 * log SigmaSFR + 1.56

    Parameters
    ----------
    SigmaSFR : str
        SFR surfance density in M_sun yr^-1 kpc^-2

    Returns
    -------
    sigma_ion_f_O22 : str
        ionised gas velocity dispersion in km s^-1

    '''
    log_SigmaSFR_f_O22 = np.log10(SigmaSFR)
    log_sigma_mol = 0.1804 * log_SigmaSFR_f_O22 + 1.56
    sigma_ion_f_O22 = np.sqrt(np.power(10,log_sigma_mol)**2 + 15**2)
    return sigma_ion_f_O22
