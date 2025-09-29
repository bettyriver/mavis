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
import astropy.units as u

def fit_cube(data_cube, snr_cube, z, lsf_fwhm_angstrom,savepath=None, 
             snr_lim=None,dx_arcsec=0.05):
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
    dx_arcsec: float.
        the pixel size in arcsec, the default is 0.05 (MAVIS 50 mas)

    Returns
    -------
    reduced_data : astropi.io.fits
        DESCRIPTION.

    '''
    
    flux_data = data_cube[0].data 
    # the unit should be per angstrom to ensure the consistency of the unit
    flux_data = flux_data * 1e-4
    # multiply by 50 milliarcsec * 50 milliarcsec = 0.05 * 0.05 arcsec * arcsec
    flux_data = flux_data * dx_arcsec * dx_arcsec
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
            guess = [np.log(max(spectrum) - min(spectrum)), 
                     wavelengths_dered_ang[np.argmax(spectrum)], 
                     2]
            # lower bounds: amplitudes > 0, sigmas > 0
            #lower_bounds = [-50,  min(wavelengths_dered_ang)-1, -np.inf]
            #upper_bounds = [-30,  max(wavelengths_dered_ang)+1, np.inf]
            lower_bounds = [-np.inf,  -np.inf, -np.inf]
            upper_bounds = [np.inf,  np.inf, np.inf]
            popt, pcov = curve_fit(gaussian,wavelengths_dered_ang,
                                   spectrum, p0=guess,#bounds=(lower_bounds, upper_bounds),
                                   maxfev=10000)
            print(popt)
            lnamp, mu, sigma = popt
            amp = np.exp(lnamp)
            flux_i = amp * (np.sqrt(2*np.pi*sigma**2))
            rel_lambda = mu / ha_wave
            velocity_i = (rel_lambda - 1) * C #km/s
            
            
            if np.abs(sigma)<lsf_sigma_angstrom_dered:
                #sigma_intrinsic = -999
                vdisp_i = 0
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

def fit_cube_twogaussian(data_cube, snr_cube, z, lsf_fwhm_angstrom,savepath=None, 
             snr_lim=None,dx_arcsec=0.05):
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
    dx_arcsec: float.
        the pixel size in arcsec, the default is 0.05 (MAVIS 50 mas)

    Returns
    -------
    reduced_data : astropi.io.fits
        DESCRIPTION.

    '''
    
    flux_data = data_cube[0].data 
    # the unit should be per angstrom to ensure the consistency of the unit
    flux_data = flux_data * 1e-4
    # multiply by 50 milliarcsec * 50 milliarcsec = 0.05 * 0.05 arcsec * arcsec
    flux_data = flux_data * dx_arcsec * dx_arcsec
    nw, ny, nx = flux_data.shape
    snr_data = snr_cube[0].data
    
    flux1_map = np.zeros((ny,nx))
    sigma1_map = np.zeros((ny,nx))
    vel1_map = np.zeros((ny,nx))
    flux2_map = np.zeros((ny,nx))
    sigma2_map = np.zeros((ny,nx))
    vel2_map = np.zeros((ny,nx))
    
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
                    flux1_map[yy,xx] = np.nan 
                    sigma1_map[yy,xx] = np.nan 
                    vel1_map[yy,xx] = np.nan 
                    flux2_map[yy,xx] = np.nan 
                    sigma2_map[yy,xx] = np.nan 
                    vel2_map[yy,xx] = np.nan 
                    continue
            
            # Initial guess: amplitude1, amp2, center1,centre2, width1, wid2
            guess = [np.log(max(spectrum) - min(spectrum))-5, 
                     np.log(max(spectrum) - min(spectrum))-5, 
                     wavelengths_dered_ang[np.argmax(spectrum)], 
                     wavelengths_dered_ang[np.argmax(spectrum)], 
                     2, 2]
            # lower bounds: amplitudes > 0, sigmas > 0
            lower_bounds = [-50, -50, min(wavelengths_dered_ang)-1, min(wavelengths_dered_ang)-1, 0.5, 0.5]
            upper_bounds = [-30, -30, max(wavelengths_dered_ang)+1, max(wavelengths_dered_ang)+1, 15, 15]

            popt, pcov = curve_fit(two_gaussian,wavelengths_dered_ang,
                                   spectrum, p0=guess,#bounds=(lower_bounds, upper_bounds),
                                   maxfev=30000)
            
            lnamp1,lnamp2, mu1,mu2, sigma1,sigma2 = popt
            amp1 = np.exp(lnamp1)
            amp2 = np.exp(lnamp2)
            flux1_i = amp1 * (np.sqrt(2*np.pi*sigma1**2))
            rel_lambda1 = mu1 / ha_wave
            velocity1_i = (rel_lambda1 - 1) * C #km/s
            
            flux2_i = amp2 * (np.sqrt(2*np.pi*sigma2**2))
            rel_lambda2 = mu2 / ha_wave
            velocity2_i = (rel_lambda2 - 1) * C #km/s
            
            
            if np.abs(sigma1)<lsf_sigma_angstrom_dered:
                #sigma_intrinsic = -999
                vdisp1_i = 0
            else:
                sigma1_intrinsic = np.sqrt(sigma1 **2 - lsf_sigma_angstrom_dered **2)
            
            
                vdisp1_i = sigma1_intrinsic / ha_wave * C # km/s, this didn't remove lsf
            if np.abs(sigma2)<lsf_sigma_angstrom_dered:
                #sigma_intrinsic = -999
                vdisp2_i = 0
            else:
                sigma2_intrinsic = np.sqrt(sigma2 **2 - lsf_sigma_angstrom_dered **2)
            
            
                vdisp2_i = sigma2_intrinsic / ha_wave * C # km/s, this didn't remove lsf
            
            # make sure assign flux 1 to larger one
            if flux1_i > flux2_i:
                flux1_map[yy,xx] = flux1_i
                sigma1_map[yy,xx] = vdisp1_i
                vel1_map[yy,xx] = velocity1_i
                
                flux2_map[yy,xx] = flux2_i
                sigma2_map[yy,xx] = vdisp2_i
                vel2_map[yy,xx] = velocity2_i
            else:
                flux1_map[yy,xx] = flux2_i
                sigma1_map[yy,xx] = vdisp2_i
                vel1_map[yy,xx] = velocity2_i
                
                flux2_map[yy,xx] = flux1_i
                sigma2_map[yy,xx] = vdisp1_i
                vel2_map[yy,xx] = velocity1_i
                
    
    # Create Primary HDU (can be empty or contain data)
    primary_hdu = fits.PrimaryHDU()
    
    # Create Image HDUs for each array
    hdu1 = fits.ImageHDU(data=flux1_map, name='Flux1')
    hdu2 = fits.ImageHDU(data=vel1_map, name='Velocity1')
    hdu3 = fits.ImageHDU(data=sigma1_map, name='Velocity dispersion1')
    hdu4 = fits.ImageHDU(data=flux2_map, name='Flux2')
    hdu5 = fits.ImageHDU(data=vel2_map, name='Velocity2')
    hdu6 = fits.ImageHDU(data=sigma2_map, name='Velocity dispersion2')
    
    # Combine into an HDUList and write to file
    hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3,hdu4, hdu5, hdu6])
    
    if savepath is not None:
    
        hdul.writeto(savepath, overwrite=True)

    
    
    return hdul

#    gaussian = amp / (np.sqrt(2*np.pi*sigma_sq)) * \
#        np.exp(-(self.w - wave_cen)**2 / (2 * sigma_sq))

def compare_one_two_gaussian_fit(data_cube, snr_cube, z, lsf_fwhm_angstrom,savepath=None, 
             snr_lim=None,dx_arcsec=0.05,variance_cube=None):
    
    
    flux_data = data_cube[0].data 
    # the unit should be per angstrom to ensure the consistency of the unit
    flux_data = flux_data * 1e-4
    # multiply by 50 milliarcsec * 50 milliarcsec = 0.05 * 0.05 arcsec * arcsec
    flux_data = flux_data * dx_arcsec * dx_arcsec
    
    if variance_cube is not None:
        variance_data = variance_cube[0].data
        variance_data = variance_data * 1e-8
        variance_data = variance_data * (dx_arcsec * dx_arcsec)**2
        
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
    
    
    x_list = []
    y_list = []
    g1_flux = []
    g1_v = []
    g1_sigma = []
    
    g2_flux1 = []
    g2_v1 = []
    g2_sigma1 = []
    g2_flux2 = []
    g2_v2 = []
    g2_sigma2 = []    
    
    snr_list = []
    
    bic_one = []
    bic_two = []
    for yy in range(ny):
        for xx in range(nx):
            spectrum = flux_data[:,yy,xx]
            variance_i = variance_data[:,yy,xx]
            snr_xy = np.max(snr_data[:,yy,xx])
            
            if snr_lim is not None:
                if snr_xy < snr_lim:
                    flux_map[yy,xx] = np.nan 
                    sigma_map[yy,xx] = np.nan 
                    vel_map[yy,xx] = np.nan 
                    continue
            
            #### one gaussian ######
            
            # Initial guess: amplitude, center, width
            guess = [np.log(max(spectrum) - min(spectrum)), 
                     wavelengths_dered_ang[np.argmax(spectrum)], 
                     2]
            # lower bounds: amplitudes > 0, sigmas > 0
            #lower_bounds = [-50,  min(wavelengths_dered_ang)-1, -np.inf]
            #upper_bounds = [-30,  max(wavelengths_dered_ang)+1, np.inf]
            lower_bounds = [-np.inf,  -np.inf, -np.inf]
            upper_bounds = [np.inf,  np.inf, np.inf]
            popt, pcov = curve_fit(gaussian,wavelengths_dered_ang,
                                   spectrum, p0=guess,#bounds=(lower_bounds, upper_bounds),
                                   maxfev=10000)
            
            lnamp, mu, sigma = popt
            amp = np.exp(lnamp)
            flux_i = amp * (np.sqrt(2*np.pi*sigma**2))
            rel_lambda = mu / ha_wave
            velocity_i = (rel_lambda - 1) * C #km/s
            
            if np.abs(sigma)<lsf_sigma_angstrom_dered:
                #sigma_intrinsic = -999
                vdisp_i = 0
            else:
                sigma_intrinsic = np.sqrt(sigma **2 - lsf_sigma_angstrom_dered **2)
            
            
                vdisp_i = sigma_intrinsic / ha_wave * C # km/s, this didn't remove lsf
            
            BIC_onegaussian = np.sum((spectrum - gaussian(wavelengths_dered_ang,lnamp,mu,sigma))**2/variance_i) + 3*np.log(len(wavelengths_dered_ang))
            
            #### two gaussians ########
            
            # Initial guess: amplitude1, amp2, center1,centre2, width1, wid2
            guess = [np.log(max(spectrum) - min(spectrum))-5, 
                     np.log(max(spectrum) - min(spectrum))-5, 
                     wavelengths_dered_ang[np.argmax(spectrum)], 
                     wavelengths_dered_ang[np.argmax(spectrum)], 
                     2, 2]
            # lower bounds: amplitudes > 0, sigmas > 0
            lower_bounds = [-70, -70, min(wavelengths_dered_ang)-1, min(wavelengths_dered_ang)-1, 0.5, 0.5]
            upper_bounds = [-30, -30, max(wavelengths_dered_ang)+1, max(wavelengths_dered_ang)+1, 15, 15]

            popt, pcov = curve_fit(two_gaussian,wavelengths_dered_ang,
                                   spectrum, p0=guess,#bounds=(lower_bounds, upper_bounds),
                                   maxfev=30000)
            
            lnamp1,lnamp2, mu1,mu2, sigma1,sigma2 = popt
            amp1 = np.exp(lnamp1)
            amp2 = np.exp(lnamp2)
            flux1_i = amp1 * (np.sqrt(2*np.pi*sigma1**2))
            rel_lambda1 = mu1 / ha_wave
            velocity1_i = (rel_lambda1 - 1) * C #km/s
            
            flux2_i = amp2 * (np.sqrt(2*np.pi*sigma2**2))
            rel_lambda2 = mu2 / ha_wave
            velocity2_i = (rel_lambda2 - 1) * C #km/s
            
            if np.abs(sigma1)<lsf_sigma_angstrom_dered:
                #sigma_intrinsic = -999
                vdisp1_i = 0
            else:
                sigma1_intrinsic = np.sqrt(sigma1 **2 - lsf_sigma_angstrom_dered **2)
            
            
                vdisp1_i = sigma1_intrinsic / ha_wave * C # km/s, this didn't remove lsf
            if np.abs(sigma2)<lsf_sigma_angstrom_dered:
                #sigma_intrinsic = -999
                vdisp2_i = 0
            else:
                sigma2_intrinsic = np.sqrt(sigma2 **2 - lsf_sigma_angstrom_dered **2)
            
            
                vdisp2_i = sigma2_intrinsic / ha_wave * C # km/s, this didn't remove lsf
            
            BIC_twogaussian = np.sum((spectrum - two_gaussian(wavelengths_dered_ang,lnamp1,lnamp2,mu1,mu2,sigma1,sigma2))**2/variance_i) + 6*np.log(len(wavelengths_dered_ang))
            
            
            fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

            # Top: One Gaussian
            axes[0].plot(wavelengths_dered_ang, spectrum, label='Data')
            axes[0].plot(wavelengths_dered_ang, gaussian(wavelengths_dered_ang,lnamp,mu,sigma), '-', lw=2, label='One-Gaussian fit')
            axes[0].set_ylabel('Flux')
            axes[0].set_title(
                f'One Gaussian: μ={mu:.2f}Å, σ={sigma:.2f}Å, '
                f'flux={flux_i:.2e}, v={velocity_i:.1f} km/s,BIC={BIC_onegaussian:.2e}'
            )
            axes[0].legend()
            
            # Bottom: Two Gaussians
            axes[1].plot(wavelengths_dered_ang, spectrum, label='Data')
            axes[1].plot(wavelengths_dered_ang, gaussian(wavelengths_dered_ang,lnamp1,mu1,sigma1), '--', lw=1.5, label='Component 1')
            axes[1].plot(wavelengths_dered_ang, gaussian(wavelengths_dered_ang,lnamp2,mu2,sigma2), ':', lw=1.5, label='Component 2')
            axes[1].plot(wavelengths_dered_ang, two_gaussian(wavelengths_dered_ang,lnamp1,lnamp2,mu1,mu2,sigma1,sigma2), '-', lw=2, label='Sum fit')
            axes[1].set_xlabel('Wavelength (Å)')
            axes[1].set_ylabel('Flux')
            axes[1].set_title(
                'Two Gaussians: '
                f'μ1={mu1:.2f}Å, σ1={sigma1:.2f}Å, flux1={flux1_i:.2e}, v1={velocity1_i:.1f} km/s | '
                f'μ2={mu2:.2f}Å, σ2={sigma2:.2f}Å, flux2={flux2_i:.2e}, v2={velocity2_i:.1f} km/s,BIC={BIC_twogaussian:.2e}'
            )
            axes[1].legend()
            
            plt.tight_layout()
            plt.savefig(savepath+str(xx)+'-'+str(yy)+'.png',dpi=200)
            plt.show()
            
            snr_list.append(snr_xy)
            
            x_list.append(xx)
            y_list.append(yy)
            g1_flux.append(flux_i)
            g1_v.append(velocity_i)
            g1_sigma.append(vdisp_i)
            bic_one.append(BIC_onegaussian)
            bic_two.append(BIC_twogaussian)
            
            if flux1_i >= flux2_i:
                g2_flux1.append(flux1_i)
                g2_v1.append(velocity1_i)
                g2_sigma1.append(vdisp1_i)
            
                g2_flux2.append(flux2_i)
                g2_v2.append(velocity2_i)
                g2_sigma2.append(vdisp2_i)
            else:
                g2_flux1.append(flux2_i)
                g2_v1.append(velocity2_i)
                g2_sigma1.append(vdisp2_i)
            
                g2_flux2.append(flux1_i)
                g2_v2.append(velocity1_i)
                g2_sigma2.append(vdisp1_i)
            
            
    df = pd.DataFrame({'x':x_list,
                       'y':y_list,
                       'one_gaussian_flux':g1_flux,
                       'one_gaussian_vel':g1_v,
                       'one_gaussian_sigma':g1_sigma,
                       'two_gaussian_flux1':g2_flux1,
                       'two_gaussian_vel1':g2_v1,
                       'two_gaussian_sigma1':g2_sigma1,
                       'two_gaussian_flux2':g2_flux2,
                       'two_gaussian_vel2':g2_v2,
                       'two_gaussian_sigma2':g2_sigma2,
                       'snr':snr_list,
                       'one_gaussian_BIC':bic_one,
                       'two_gaussian_BIC':bic_two
                       })
    
    return df
            


def gaussian(x, lnamp, mu, sigma):
    amp = np.exp(lnamp)
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def two_gaussian(x, lnamp1, lnamp2, mu1, mu2, sigma1, sigma2):
    amp1 = np.exp(lnamp1)
    amp2 = np.exp(lnamp2)
    return amp1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) + amp2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))

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


def arcsec_to_kpc(rad_in_arcsec,z):
    from astropy.cosmology import LambdaCDM
    lcdm = LambdaCDM(70,0.3,0.7)
    distance = lcdm.angular_diameter_distance(z).value # angular diameter distance, Mpc/radian
    rad_in_kpc = rad_in_arcsec * distance * np.pi/(180*3600)*1000
    return rad_in_kpc
    
def ha_flux_to_SigmaSFR(ha_flux_map,z,dx_arcsec,dy_arcsec):
    '''
    

    Parameters
    ----------
    ha_flux_map : 2d-array
        ha flux in the unit of erg/s/cm**2.
    z : float
        redshift.
    dx_arcsec : float
        width of pixel in x direction in arcsec.
    dy_arcsec : float
        width of pixel in y direction in arcsec.

    Returns
    -------
    SigmaSFR_map: 2d-array
        Sigma SFR map in the unit of M_sun yr^-1 kpc^-2

    '''
    
    from astropy.cosmology import LambdaCDM
    lcdm = LambdaCDM(70,0.3,0.7)
    luminosity_distance_Mpc = lcdm.luminosity_distance(z)
    luminosity_diatance_cm = luminosity_distance_Mpc.to(u.cm)
    # 4*pi*lumi_dis**2 * flux
    # note the flux should add 1e-20 to get the value in erg/s/cm**2
    luminosity_map = 4*np.pi*(luminosity_diatance_cm.value)**2*ha_flux_map
    SFR_map = luminosity_map/(1.26*1.53*1e41) # from Mun+24
    
    pixel_wid_kpc = arcsec_to_kpc(rad_in_arcsec=np.sqrt(dx_arcsec * dy_arcsec),
                                       z=z)
    
    SigmaSFR_map = SFR_map/(pixel_wid_kpc)**2
    
    return SigmaSFR_map