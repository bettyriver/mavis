#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 14:34:22 2025

fit the MSIM output datacube

@author: ymai0110
"""

import astropy.io.fits as fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fit_cube(data_cube, snr_cube, z, lsf_fwhm_angstrom,savepath=None, 
             snr_lim=None):
    '''
    

    Parameters
    ----------
    data_cube : astropy.io.fits
        fits file of _reduced_flux_cal.
    snr_cube : astropy.io.fits
        fits file of reduced SNR. it is signal to noise ratio of the reduced
        data cube per pixel
    z : float
        the redshift of this galaxy.
    lsf_fwhm_angstrom:
        the FWHM of LSF in angstrom.
    snr_lim : int or float, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    reduced_data : astropi.io.fits
        DESCRIPTION.

    '''
    
    flux_data = data_cube[0].data 
    # the unit should be per angstrom to ensure the consistency of the unit
    flux_data = flux_data * 1e-4
    # multiply by 50 milliarcsec * 50 milliarcsec = 0.05 * 0.05 arcsec * arcsec
    flux_data = flux_data * 0.050 * 0.050
    nw, ny, nx = flux_data.shape
    snr_data = snr_cube[0].data
    
    flux_map = np.zeros((ny,nx))
    sigma_map = np.zeros((ny,nx))
    vel_map = np.zeros((ny,nx))
    
    HD1 = data_cube[0].header
    wavelength_crpix = HD1['CRPIX3']
    wavelength_crval = HD1['CRVAL3']
    wavelength_del_mic = HD1['CDELT3']
    
    lsf_fwhm_angstrom_dered = lsf_fwhm_angstrom/(1 + z)
    lsf_sigma_angstrom_dered = lsf_fwhm_angstrom_dered/(2 * np.sqrt(2 * np.log(2)))
    
    # wavelength array in micrometer
    wavelengths_mic = wavelength_crval + \
        (np.arange(0,nw)+1-wavelength_crpix)*wavelength_del_mic
    
    # deredshifted wavelength array in micrometer
    wavelengths_dered = wavelengths_mic/(1+z)
    
    # deredshift wavelength array in angstrom
    wavelengths_dered_ang = wavelengths_dered * 10000
    
    ha_wave = 6563
    
    C = 2.99792458e5 # km/s
    
    for yy in range(ny):
        for xx in range(nx):
            spectrum = flux_data[:,yy,xx]
            snr_xy = np.max(snr_data[:,yy,xx])
            
            if snr_lim is not None:
                if snr_xy < snr_lim:
                    flux_map[yy,xx] = np.nan 
                    sigma_map[yy,xx] = np.nan 
                    vel_map[yy,xx] = np.nan 
                    continue
            
            # Initial guess: amplitude, center, width
            guess = [max(spectrum) - min(spectrum), 
                     wavelengths_dered_ang[np.argmax(spectrum)], 
                     0.8]
            popt, pcov = curve_fit(gaussian,wavelengths_dered_ang,
                                   spectrum, p0=guess,
                                   maxfev=10000)
            
            amp, mu, sigma = popt
            flux_i = amp * (np.sqrt(2*np.pi*sigma**2))
            rel_lambda = mu / ha_wave
            velocity_i = (rel_lambda - 1) * C #km/s
            
            
            if sigma<lsf_sigma_angstrom_dered:
                #sigma_intrinsic = -999
                vdisp_i = -999
            else:
                sigma_intrinsic = np.sqrt(sigma **2 - lsf_sigma_angstrom_dered **2)
            
            
                vdisp_i = sigma_intrinsic / ha_wave * C # km/s, this didn't remove lsf
            
            flux_map[yy,xx] = flux_i
            sigma_map[yy,xx] = vdisp_i
            vel_map[yy,xx] = velocity_i
    
    # Create Primary HDU (can be empty or contain data)
    primary_hdu = fits.PrimaryHDU()
    
    # Create Image HDUs for each array
    hdu1 = fits.ImageHDU(data=flux_map, name='Flux')
    hdu2 = fits.ImageHDU(data=vel_map, name='Velocity')
    hdu3 = fits.ImageHDU(data=sigma_map, name='Velocity dispersion')
    
    # Combine into an HDUList and write to file
    hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])
    
    if savepath is not None:
    
        hdul.writeto(savepath, overwrite=True)

    
    
    return hdul


#    gaussian = amp / (np.sqrt(2*np.pi*sigma_sq)) * \
#        np.exp(-(self.w - wave_cen)**2 / (2 * sigma_sq))

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def centre_pix(x_range,y_range,nx, ny, centre_coor):
    '''
    

    Parameters
    ----------
    x_range : tuple
        (x_min,x_max).
    y_range : tuple
        (y_min,y_max).
    nx: int
        number of x pixels
    ny: int
        number of y pixels
    centre_coor : tuple
        (x_cen,y_cen).

    Returns
    -------
    x_pix, y_pix
    note: pix indext start from 0

    '''
    
    x_min, x_max = x_range  # X range
    y_min, y_max = y_range  # Y range
    
    
    delta_x = (x_max - x_min)/nx
    delta_y = (y_max - y_min)/ny
    
    x_c, y_c = centre_coor
    
    x_pix = (x_c - (x_min + 0.5*delta_x))/delta_x
    y_pix = (y_c - (y_min + 0.5*delta_y))/delta_y
    
    return x_pix, y_pix
    
    