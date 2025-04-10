#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:28:12 2025


construct 3d data cube base on given blobs (flux distribution) and velocity
profile.

@author: ymai0110
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

class cmap:
    flux = 'inferno'
    v = 'RdYlBu_r'
    vdisp = 'YlOrBr'
    residuals = 'RdYlBu_r'

class Construct_cube:
    
    def __init__(
            self, blobs_df, global_param_df, x_range, y_range, w_range,
            nx, ny,redshift, lsf_fwhm=0.8):
        '''
        Parameters
        ----------
        blobs_df : pandas.DataFrame
            A dataframe that includes RC, THETAC, W, Q, PHI, FLUX0 of all blobs.
            It's a output of post_blobby3d. We assume Q=1 at this stage to simplify
            the function.
        global_param_df: pandas.DataFrame
            the global parameter from blobby3d
        x_range: tuple
            The range of x. It's in the metadata of blobby3d.
        y_range: tuple
            The range of y. It's in the metadata of blobby3d.
        w_range: tuple
            Thane range of wavelength. It's in the metadata of blobby3d.
        nx : int
            The number of pixel in x direction.
        ny : int
            The number of pixel in y direction.
        lsf_fwhm : float
            the fwhm of lsf in Angstrom. 
            default is 0.8, equvalent to R=10000
            (i.e. MAVIS ideal spectral resolution)
            It will determine the wavelength pixel
            size. dw = 1/2 * lsf_fwhm, wavelength pixel size
        redshift: float
            The redshift for this galaxy
        lsf_fwhm:
            The fwhm of line-spread-function.
            
        '''
        self.blobs_df = blobs_df
        
        self.blobs_df['X'] = blobs_df['RC'] * np.cos(blobs_df['THETAC'])
        self.blobs_df['Y'] = blobs_df['RC'] * np.sin(blobs_df['THETAC'])
        
        # the global parameter from blobby3d, take the first sample from it
        self.global_param_df = global_param_df
        
        # the first sample
        sample = 0
        
        # coordinate of galaxy centre
        self.XC = self.global_param_df.iloc[sample]['XC']
        self.YC = self.global_param_df.iloc[sample]['YC']
        
        # velocity profile parameter 'VSYS', 'VMAX', 'VSLOPE', 'VGAMMA', 'VBETA'
        self.VSYS = self.global_param_df.iloc[sample]['VSYS']
        self.VMAX = self.global_param_df.iloc[sample]['VMAX']
        self.VSLOPE = self.global_param_df.iloc[sample]['VSLOPE']
        self.VGAMMA = self.global_param_df.iloc[sample]['VGAMMA']
        self.VBETA = self.global_param_df.iloc[sample]['VBETA']
        
        # pa
        self.PA = self.global_param_df.iloc[sample]['PA']
        # inclination
        self.INC = self.global_param_df.iloc[sample]['INC']
        
        self.VDISP0 = self.global_param_df.iloc[sample]['VDISP0']
        # velocity dispersion in the unit of km/s
        self.vdisp = np.exp(self.VDISP0) 
        
        self.x_range = x_range
        self.y_range = y_range
        self.w_range = w_range
        
        # Define flux map properties
        self.x_min, self.x_max = x_range  # X range
        self.y_min, self.y_max = y_range  # Y range
        self.w_min, self.w_max = w_range  # W range
        self.nx = nx
        self.ny = ny
        #self.nw = nw
        
        
        lsf_fwhm_deredshift = lsf_fwhm/(1 + redshift)
        self.lsf_fwhm_deredshift = lsf_fwhm_deredshift
        self.lsf_fwhm = lsf_fwhm
        temp_dw = 0.5 * lsf_fwhm_deredshift
        self.nw = int((self.w_max - self.w_min)/temp_dw + 1)
        
        
        # Generate pixel grid
        self.x = np.linspace(self.x_min, self.x_max, nx)
        self.y = np.linspace(self.y_min, self.y_max, ny)
        self.w = np.linspace(self.w_min, self.w_max, self.nw)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        self.dx = (self.x_max - self.x_min)/nx
        self.dy = (self.y_max - self.y_min)/ny
        self.dw = (self.w_max - self.w_min)/self.nw
        
        # speed of light in km/s
        self.C = 2.99792458e5
        
        self.pixel_width = np.sqrt(self.nx * self.ny)
        self.redshift = redshift

    def make_flux_map(self):
        '''
        Take the blobs dataframe, return the 2-d flux map with given grid-size.
        18Mar25: add PA and inclination
        
        Todo:
            1. DONE: take the inclination of the galaxy into account
            2. consider Q and PHI of blobs
            3. consider ovesample case, i.e. blob size is smaller than pixel size
                The flux map may be inaccurate if the blobs are oversample
                However, I don't fully understand DiscModel line 695 amp.
                Neither the LookupExp, which could be cumulative function....
                
        
        Parameters
        ----------
        
    
        Returns
        -------
        flux_map: 2-d array
            Flux map.
        '''
        sin_pa = np.sin(self.PA)
        cos_pa = np.cos(self.PA)
        cos_inc = np.cos(self.INC)
        invcos_inc = 1.0/cos_inc
        
        
        flux_map = np.zeros((self.ny, self.nx))
        
        
        # get rotated/inc disc coordinate, the 'd' means disc
        ## shift
        xd_shft = self.X - self.XC
        yd_shft = self.Y - self.YC
        
        ## rotate by pa arount z (counter-clockwise, East pa=0)
        xd_rot = xd_shft*cos_pa + yd_shft*sin_pa
        yd_rot = -xd_shft*sin_pa + yd_shft*cos_pa
        
        ## rotate by inclination
        yd_rot *= invcos_inc
        
        
        # for each blob, generate the flux map
        # flux distribution of each blob is describe by (eq. 6 in Varidel+2019)
        # F = f/(2*pi*w**2)*exp(-(x**2+y**2)/(2*w**2))
        # f is the amplitude, w is the width of the blob, x, y are the distance
        # from the center of the blob
        
        for _, row in self.blobs_df.iterrows():
            
            
            
            
            
            blob_x, blob_y = row['X'], row['Y']
            blob_w = row['W']
            blob_flux = row['FLUX0']
            
            # Compute distances from blob center
            X_b = xd_rot - blob_x
            Y_b = yd_rot - blob_y
            
            # multiply by the size of each pixel to get flux in that pixel
            amp = self.dx * self.dy * blob_flux / (2 * np.pi * blob_w**2 * cos_inc)
            
            # Compute Gaussian flux distribution
            blob_flux_distribution = amp * \
                        np.exp(-(X_b**2 + Y_b**2) / (2 * blob_w**2))
            
            # Add contribution to flux map
            flux_map += blob_flux_distribution
        
        # make flux map show the integrate flux of that spaxel
        # note this is a simlified calculation, the accurate one need to use
        # accumulative function
        
        
        
        return flux_map
    
        
        
    def plot_flux_map(self,vmin=-2,vmax=1):
        '''
        plot the flux map

        Parameters
        ----------
        vmin : float, optional
            min for the flux map. The default is -2.
        vmax : float, optional
            max for the flux map. The default is 1.

        Returns
        -------
        None.

        '''
        
        
        flux_map = self.make_flux_map()
        
        # Define flux map properties
        x_min, x_max = self.x_range  # X range
        y_min, y_max = self.y_range  # Y range
        
        # Plot the flux map
        plt.figure(figsize=(6, 5))
        plt.imshow(np.log10(flux_map), extent=[x_min, x_max, y_min, y_max], origin='lower', 
                   cmap='inferno',vmin=vmin,vmax=vmax)
        plt.colorbar(label='log10(Flux)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Flux Map from Blobs in DataFrame')
        plt.show()
        
        
    def make_vel_map(self):
        '''
        Take the velocity map parameter from blobby3d and make velocity map.
        
        Parameters
        ----------
        
    
        Returns
        -------
        None.
    
        '''
        sin_pa = np.sin(self.PA)
        cos_pa = np.cos(self.PA)
        cos_inc = np.cos(self.INC)
        invcos_inc = 1.0/cos_inc
        sin_inc = np.sin(self.INC)
        
        # get rotated/inc disc coordinate, the 'd' means disc
        ## shift
        xd_shft = self.X - self.XC
        yd_shft = self.Y - self.YC
        
        ## rotate by pa arount z (counter-clockwise, East pa=0)
        xd_rot = xd_shft*cos_pa + yd_shft*sin_pa
        yd_rot = -xd_shft*sin_pa + yd_shft*cos_pa
        
        ## rotate by inclination
        yd_rot *= invcos_inc
        
        # calculate radius
        rad = np.sqrt(xd_rot**2 + yd_rot**2)
        
        angle = np.arctan2(yd_rot, xd_rot)
        cos_angle = np.cos(angle)
        
        # build the velocity map
        
        velocity_map = self.VMAX * np.power(1.0 + self.VSLOPE/rad, self.VBETA)
        velocity_map /= np.power(
            1.0 + np.power(self.VSLOPE/rad, self.VGAMMA), 1.0/self.VGAMMA)
        velocity_map *= sin_inc * cos_angle
        
        velocity_map += self.VSYS
        
        
        return velocity_map
    
    def plot_vel_map(self):
        '''
        plot the velocity map

        Returns
        -------
        None.

        '''
        vel_map = self.make_vel_map()
        
        # Define flux map properties
        x_min, x_max = self.x_range  # X range
        y_min, y_max = self.y_range  # Y range
        
        # Plot the flux map
        plt.figure(figsize=(6, 5))
        plt.imshow(vel_map, extent=[x_min, x_max, y_min, y_max], origin='lower', 
                   cmap=cmap.v)
        plt.colorbar(label='velocity')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('velocity Map')
        plt.show()

    def make_rel_lambda_map(self):
        '''
        calculate the factor for the lambda shift

        Returns
        -------
        rel_lambda : 2-d np.array
            the factor for the lambda shift.

        '''
        velocity_map = self.make_vel_map()
        
        # calculate the relative wavelength
        rel_lambda = velocity_map/self.C + 1 
        
        return rel_lambda
    
    def make_vdisp_map(self,vdisp=None):
        '''
        make a constant velocity dispersion map

        Parameters
        ----------
        vdisp : float
            gas velocity dispersion in unit of km/s. if vidsp is None,
            take the vdisp from blobb3d parameter

        Returns
        -------
        vdisp_map : 2-d np.array
            velocity dispersion map

        '''
        if vdisp is None:
            vdisp = self.vdisp
            
        
        
        vdisp_map = np.ones((self.ny, self.nx)) * vdisp
        
        return vdisp_map
        
    
    def make_cube(self,hsimcube=False):
        '''
        make the 3d datacube, the default data cube have the same units as
        the MAGPI datacube, i.e. the flux unit is erg/s/cm^2/AA.

        Parameters
        ----------
        hsimcube : boolen, optional
            If True, the flux is divided by the spaxel scale in arcsec, i.e.
            the unit of flux becomes erg/s/cm^2/AA/arcsec^2.
            The default is False.

        Returns
        -------
        cube : 3-d array
            DESCRIPTION.

        '''
        
        
        flux_map = self.make_flux_map()
        # for HSIM cube, The flux density units for input datacubes require 
        # the usual flux (e.g., erg/s/cm^2/AA) to be divided by the spaxel 
        # scale in arcsec. This then gives e.g. erg/s/cm^2/AA/arcsec^2.
        if hsimcube:
            flux_map = flux_map / (self.dx * self.dy) 
        rel_lambda_map = self.make_rel_lambda_map()
        vdisp_map = self.make_vdisp_map()
        
        vdisp_c_map = vdisp_map/self.C
        
        ha_wave = 6563
        
        # 10km/s pixel
        
        
        cube = np.zeros((self.nw,self.ny,self.nx))
        # the below shape is wrong!! although I don't fully understand why 
        # the shape is the above...
        # cube = np.zeros((self.nx,self.ny,self.nw))
        
        for i in range(self.ny):
            for j in range(self.nx):
                amp = flux_map[i,j]
                wave_cen = ha_wave*rel_lambda_map[i,j]
                sigma = ha_wave * vdisp_c_map[i,j]
                
                #gaussian = np.zeros(self.nw)
                # as all calculation here are based on deredshift spectrum
                # so use deredshift fwhm
                sigma_lsf =  self.lsf_fwhm_deredshift/ (2 * np.sqrt(2 * np.log(2)))
                
                
                sigma_sq = sigma**2 + sigma_lsf**2
                
                
                gaussian = amp / (np.sqrt(2*np.pi*sigma_sq)) * \
                    np.exp(-(self.w - wave_cen)**2 / (2 * sigma_sq))
                
                
                
                
                
                cube[:,i,j] = gaussian
                
        
        
        return cube
        
        
        
    def make_fits(self,savepath,hsimcube=False):
        
        data = self.make_cube(hsimcube=hsimcube)
        
        
        # Create a primary HDU
        hdu = fits.PrimaryHDU(data)
        
        # Modify the header
        hdu.header['OBJECT'] = 'Example Data'
        
        # spatial info
        hdu.header['NAXIS1'] = self.nx
        hdu.header['NAXIS2'] = self.ny
        
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC---TAN'
        
        hdu.header['CDELT1'] = self.dx
        hdu.header['CDELT2'] = self.dy
        
        hdu.header['CUNIT1'] = 'deg'
        hdu.header['CUNIT2'] = 'deg'
        
        # spectral info
        hdu.header['NAXIS3'] = self.nw
        hdu.header['CTYPE3'] = 'wavelength'
        
        # recover the redshifted wavelength
        hdu.header['CDELT3'] = self.dw * (1 + self.redshift)
        
        hdu.header['SPECRES'] = self.lsf_fwhm  #lsf fwhm in Angstrom
        
        hdu.header['CRPIX3'] = 1
        hdu.header['CRVAL3'] = (self.w_min + 1/2 * self.dw) * (1 + self.redshift)
        hdu.header['CUNIT3'] = 'Angstrom'
        
        # Flux unit
        if hsimcube:
            hdu.header['BUNIT'] = '10**(-20)*erg/s/cm**2/Angstrom/arcsec**2'
        else:
            hdu.header['BUNIT'] = '10**(-20)*erg/s/cm**2/Angstrom'
        
        
        # Create an HDU list
        hdul = fits.HDUList([hdu])
        
        # Write to a FITS file
        hdul.writeto(savepath, overwrite=True)
        