#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 13:50:34 2025

@author: ymai0110
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:28:12 2025


construct 3d data cube base on given blobs (flux distribution) and velocity
profile.

update: construct cube blob by blob, allowing each cube attach one blob properties
        and higher flexibility on blob properties

@author: ymai0110
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from scipy.stats import norm
from SigmaSFR_to_sigma_models import K18_F_and_T, K18_F_only, O22_F_only, K18_F_and_T_quick

class cmap:
    flux = 'inferno'
    v = 'RdYlBu_r'
    vdisp = 'YlOrBr'
    residuals = 'RdYlBu_r'

class Construct_cube:
    
    def __init__(
            self, blobs_df, global_param_df, x_range, y_range, w_range,
            nx, ny,redshift, lsf_fwhm=0.8,mask=None):
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
        mask: 2-d array
            mask from original datacube. 1 to mask, 0 to keep
            
        '''
        self.blobs_df = blobs_df.copy()
        
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
        
        # the range is the outmost range
        self.dx = (self.x_max - self.x_min)/ (nx)
        self.dy = (self.y_max - self.y_min)/ (ny)
        self.dw = (self.w_max - self.w_min)/ (self.nw)
        
        # coordinate of pixel is from the center of pixel
        # Generate pixel grid
        self.x = np.linspace(self.x_min+0.5*self.dx, self.x_max-0.5*self.dx, nx)
        self.y = np.linspace(self.y_min+0.5*self.dy, self.y_max-0.5*self.dy, ny)
        self.w = np.linspace(self.w_min+0.5*self.dw, self.w_max-0.5*self.dw, self.nw)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        
        
        
        # speed of light in km/s
        self.C = 2.99792458e5
        
        self.pixel_width = np.sqrt(self.nx * self.ny)
        self.redshift = redshift
        
        
        self.mask = mask
        
        
        
        sin_pa = np.sin(self.PA)
        cos_pa = np.cos(self.PA)
        cos_inc = np.cos(self.INC)
        invcos_inc = 1.0/cos_inc
        
        
        
        #### work with mask 
        if self.mask is not None:
            
            mask_ny, mask_nx = self.mask.shape
            
            # the range is the outmost range
            mask_dx = (self.x_max - self.x_min)/ (mask_nx)
            mask_dy = (self.y_max - self.y_min)/ (mask_ny)
            
            # Generate pixel grid
            mask_x = np.linspace(self.x_min+0.5*mask_dx, self.x_max-0.5*mask_dx, mask_nx)
            mask_y = np.linspace(self.y_min+0.5*mask_dy, self.y_max-0.5*mask_dy, mask_ny)
            
            mask_X, mask_Y = np.meshgrid(mask_x, mask_y)
            
            
            
            # get rotated/inc disc coordinate, the 'd' means disc
            ## shift
            mask_xd_shft = mask_X - self.XC
            mask_yd_shft = mask_Y - self.YC
            
            
            ## rotate by pa arount z (counter-clockwise, East pa=0)
            self.mask_xd_rot = mask_xd_shft*cos_pa + mask_yd_shft*sin_pa
            self.mask_yd_rot = -mask_xd_shft*sin_pa + mask_yd_shft*cos_pa
            
            ## rotate by inclination
            self.mask_yd_rot *= invcos_inc
            
    def make_one_blob_flux_map(self,blob_row):
        
        sin_pa = np.sin(self.PA)
        cos_pa = np.cos(self.PA)
        cos_inc = np.cos(self.INC)
        invcos_inc = 1.0/cos_inc
        
        blob_x, blob_y = blob_row['X'], blob_row['Y']
        blob_w = blob_row['W']
        blob_flux = blob_row['FLUX0']
        
        
        flux_map = np.zeros((self.ny, self.nx))
        
        if blob_w * cos_inc < 0.5 * np.sqrt(self.dx * self.dy):
            si = 2
        elif blob_w * cos_inc < np.sqrt(self.dx * self.dy):
            si = 1
        else:
            si = 0
        
        dxfs = self.dx / (2.0*si + 1)
        dyfs = self.dy / (2.0*si + 1)
        
        # multiply by the size of each pixel to get flux in that pixel
        #amp = self.dx * self.dy * blob_flux / (2 * np.pi * blob_w**2 * cos_inc)
        amp = dxfs * dyfs * blob_flux / (2 * np.pi * blob_w**2 * cos_inc)
        
        for xi in range(-si, si+1):
            for yi in range(-si, si+1):
        
                # get rotated/inc disc coordinate, the 'd' means disc
                ## shift
                xd_shft = self.X - self.XC
                yd_shft = self.Y - self.YC
                
                xd_shft = xd_shft + xi*dxfs
                yd_shft = yd_shft + yi*dyfs
                
                ## rotate by pa arount z (counter-clockwise, East pa=0)
                xd_rot = xd_shft*cos_pa + yd_shft*sin_pa
                yd_rot = -xd_shft*sin_pa + yd_shft*cos_pa
                
                ## rotate by inclination
                yd_rot *= invcos_inc
                
                
                
                
                # Compute distances from blob center
                X_b = xd_rot - blob_x
                Y_b = yd_rot - blob_y
                
                
                
                # Compute Gaussian flux distribution
                blob_flux_distribution = amp * \
                            np.exp(-(X_b**2 + Y_b**2) / (2 * blob_w**2))
                
                # Add contribution to flux map
                flux_map += blob_flux_distribution
        return flux_map
    
    def make_one_blob_velocity_map(self,blob_row):
        sin_pa = np.sin(self.PA)
        cos_pa = np.cos(self.PA)
        cos_inc = np.cos(self.INC)
        invcos_inc = 1.0/cos_inc
        sin_inc = np.sin(self.INC)
        
        blob_x, blob_y = blob_row['X'], blob_row['Y']
        blob_w = blob_row['W']
        blob_flux = blob_row['FLUX0']
        
        
        
        
        # calculate radius
        rad = np.full((self.ny, self.nx),np.sqrt(blob_x**2 + blob_y**2))
        
        angle = np.arctan2(blob_y, blob_x)
        cos_angle = np.cos(angle)
        
        # build the velocity map
        
        velocity_map = self.VMAX * np.power(1.0 + self.VSLOPE/rad, self.VBETA)
        velocity_map /= np.power(
            1.0 + np.power(self.VSLOPE/rad, self.VGAMMA), 1.0/self.VGAMMA)
        velocity_map *= sin_inc * cos_angle
        
        velocity_map += self.VSYS
        
        
        return velocity_map
        

    def make_one_blob_cube(self,blob_row):
        blob_x, blob_y = blob_row['X'], blob_row['Y']
        blob_w = blob_row['W']
        blob_flux = blob_row['FLUX0']
        
        flux_map = self.make_one_blob_flux_map(blob_row=blob_row)
        velocity_map = self.make_one_blob_velocity_map(blob_row=blob_row)
        
        # random vel of that blob in addition to rotation vel
        random_vel = np.random.normal(0,10) # km/s
        velocity_map = velocity_map + random_vel
        
        
        
        
        return None
    
    def make_all_blob_cube(self):
        
        all_blob_cube = np.zeros((self.nw,self.ny,self.nx))
        
        for _, row in self.blobs_df.iterrows():
            
            blob_x, blob_y = row['X'], row['Y']
            blob_w = row['W']
            blob_flux = row['FLUX0']
            
            
            if self.mask is not None:
                # Compute the distance to (blob_x, blob_y)
                dist = np.sqrt((self.mask_xd_rot - blob_x)**2 + (self.mask_yd_rot - blob_y)**2)
                
                # Get the index of the minimum distance
                i, j = np.unravel_index(np.argmin(dist), dist.shape)
                
                
                if self.mask[i,j] == 1:
                    continue
            
            
            all_blob_cube += self.make_all_blob_cube(blob_row=row)