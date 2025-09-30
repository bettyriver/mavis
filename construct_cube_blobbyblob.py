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
        self.ha_wave = 6563
        
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
        
    
    
    def calculate_SigmaSFR_one_blob(self,blob_row):
        blob_x, blob_y = blob_row['X'], blob_row['Y']
        blob_w = blob_row['W']
        blob_flux = blob_row['FLUX0']
        
        from astropy.cosmology import LambdaCDM
        lcdm = LambdaCDM(70,0.3,0.7)
        luminosity_distance_Mpc = lcdm.luminosity_distance(self.redshift)
        luminosity_diatance_cm = luminosity_distance_Mpc.to(u.cm)
        # 4*pi*lumi_dis**2 * flux
        # note the flux should add 1e-20 to get the value in erg/s/cm**2
        luminosity = 4*np.pi*(luminosity_diatance_cm.value)**2*blob_flux*1e-20
        SFR = luminosity/(1.26*1.53*1e41) # from Mun+24
        
        pixel_wid_kpc = self.arcsec_to_kpc(rad_in_arcsec=2*blob_w,
                                           z=self.redshift)
        
        SigmaSFR = SFR/(pixel_wid_kpc)**2
        
        return SigmaSFR
        
    
    def make_one_blob_cube(self,blob_row,sigmaSigmaSFRpaper='Wisnioski+12',
                           hsimcube=True):
        blob_x, blob_y = blob_row['X'], blob_row['Y']
        blob_w = blob_row['W']
        blob_flux = blob_row['FLUX0']
        
        flux_map = self.make_one_blob_flux_map(blob_row=blob_row)
        velocity_map = self.make_one_blob_velocity_map(blob_row=blob_row)
        
        # random vel of that blob in addition to rotation vel
        random_vel = np.random.normal(0,10) # km/s
        velocity_map = velocity_map + random_vel
        
        # calculate SigmaSFR for this blob
        # Maybe try SigmaSFR = SFR/(4*pi*blob_w**2) (i.e.take 2*blob_w as range of blob)
        
        SigmaSFR = self.calculate_SigmaSFR_one_blob(blob_row=blob_row)
        
        vdisp = self.vdisp_from_SigmaSFR(SigmaSFR=SigmaSFR,paper=sigmaSigmaSFRpaper)
        vdisp_map = np.full_like(velocity_map, vdisp)
        
        outflow_flag = False
        
        # outflow
        if np.log10(SigmaSFR) > -2: # i use -1.5 for the stack map. i lower the bar here
            outflow_flag = True
            outflow_v = 20
            
            #outflow vdisp
            if np.log10(SigmaSFR)>0:
                outflow_vdisp = 100
            else:
                slope = (100 - 50) / 2
                outflow_vdisp = 50 + slope * (np.log10(SigmaSFR) + 2)
            
        
        ##### start construct cube
        if not outflow_flag:
            if hsimcube:
                flux_map = flux_map / (self.dx * self.dy) 
            
            rel_lambda_map = velocity_map/self.C + 1 
            vdisp_c_map = vdisp_map/self.C
            cube = np.zeros((self.nw,self.ny,self.nx))
            for i in range(self.ny):
                for j in range(self.nx):
                    amp = flux_map[i,j]
                    wave_cen = self.ha_wave*rel_lambda_map[i,j]
                    sigma = self.ha_wave * vdisp_c_map[i,j]
                    
                    #gaussian = np.zeros(self.nw)
                    # as all calculation here are based on deredshift spectrum
                    # so use deredshift fwhm
                    sigma_lsf =  self.lsf_fwhm_deredshift/ (2 * np.sqrt(2 * np.log(2)))
                    
                    
                    sigma_sq = sigma**2 + sigma_lsf**2
                    
                    
                    #gaussian = amp / (np.sqrt(2*np.pi*sigma_sq)) * \
                    #    np.exp(-(self.w - wave_cen)**2 / (2 * sigma_sq))
                    
                    w_cdf = np.concatenate((self.w,[self.w[-1]+self.dw]))
                    w_cdf = w_cdf - self.dw/2
                    
                    cdf_scipy = norm.cdf(w_cdf, wave_cen, np.sqrt(sigma_sq))
                    
                    gaussian = np.diff(cdf_scipy)/self.dw * amp
                    
                    cube[:,i,j] = gaussian
        
        else: # for the case with outflow
            if hsimcube:
                flux_map = flux_map / (self.dx * self.dy) 
            
            rel_lambda_map = velocity_map/self.C + 1 
            vdisp_c_map = vdisp_map/self.C
            cube = np.zeros((self.nw,self.ny,self.nx))
            outflow_velocity_map = np.full_like(velocity_map, outflow_v)
            
            outflow_rel_lambda_map = (velocity_map - outflow_velocity_map)/self.C + 1 
            outflow_vdisp_c_map = np.full_like(velocity_map,outflow_vdisp)/self.C
            
            cube = np.zeros((self.nw,self.ny,self.nx))
            for i in range(self.ny):
                for j in range(self.nx):
                    
                    logSigmaSFR = np.log10(SigmaSFR)
                    
                    
                    # main flux
                    amp = 0.6 * flux_map[i,j]
                    wave_cen = self.ha_wave*rel_lambda_map[i,j]
                    sigma = self.ha_wave * vdisp_c_map[i,j]
                    
                    sigma_lsf =  self.lsf_fwhm_deredshift/ (2 * np.sqrt(2 * np.log(2)))
                    
                    
                    sigma_sq = sigma**2 + sigma_lsf**2
                    
                    
                    #gaussian = amp / (np.sqrt(2*np.pi*sigma_sq)) * \
                    #    np.exp(-(self.w - wave_cen)**2 / (2 * sigma_sq))
                    
                    w_cdf = np.concatenate((self.w,[self.w[-1]+self.dw]))
                    w_cdf = w_cdf - self.dw/2
                    
                    cdf_scipy = norm.cdf(w_cdf, wave_cen, np.sqrt(sigma_sq))
                    
                    gaussian = np.diff(cdf_scipy)/self.dw * amp
                    
                    # outflow flux
                    amp = 0.4 * flux_map[i,j]
                    wave_cen = self.ha_wave * outflow_rel_lambda_map[i,j]
                    sigma = self.ha_wave * outflow_vdisp_c_map[i,j]
                    sigma_lsf =  self.lsf_fwhm_deredshift/ (2 * np.sqrt(2 * np.log(2)))
                    
                    
                    sigma_sq = sigma**2 + sigma_lsf**2
                    w_cdf = np.concatenate((self.w,[self.w[-1]+self.dw]))
                    w_cdf = w_cdf - self.dw/2
                    
                    cdf_scipy = norm.cdf(w_cdf, wave_cen, np.sqrt(sigma_sq))
                    
                    gaussian = gaussian + np.diff(cdf_scipy)/self.dw * amp
                    cube[:,i,j] = gaussian
        
        if outflow_flag:
            meta = {
                    "X": blob_row["X"],
                    "Y": blob_row["Y"],
                    "W": blob_row["W"],
                    "FLUX0": blob_row["FLUX0"],
                    "velocity": velocity_map[0,0],
                    "SigmaSFR": SigmaSFR,
                    "vdisp": vdisp,
                    "outflow_velocity": outflow_v,
                    "outflow_vdisp": outflow_vdisp
                }
        else:
            meta = {
                    "X": blob_row["X"],
                    "Y": blob_row["Y"],
                    "W": blob_row["W"],
                    "FLUX0": blob_row["FLUX0"],
                    "velocity": velocity_map[0,0],
                    "SigmaSFR": SigmaSFR,
                    "vdisp": vdisp,
                    "outflow_velocity": 0,
                    "outflow_vdisp": 0
                }
        
        return cube, meta
    
    
    def vdisp_from_SigmaSFR(self,SigmaSFR,paper='Wisnioski+12'):
        '''
        calculate ionised gas velocity dispersion from SigmaSFR.

        Parameters
        ----------
        SigmaSFR: float
            SFR surface density
        paper : str, optional
            paper that relation is from. The default is 'Wisnioski+12'.
            'Wisnioski+12': relation from observation of individual HII regions
                            from E. Wisnioski+2012
            'Mai+24': relation from observation of galaxy global parameters
                        from Y. Mai+2024
            'Krumholz+18_FT': feedback+tranport model from Krumholz+2018
            'Krumholz+18_Fonly': feedback only model from Krumholz+2018
            'Ostriker+22_Fonly': feedback only model from Ostriker+2022

        Returns
        -------
        vdisp_map : TYPE
            DESCRIPTION.

        '''
        
        
        valid_options = ['Wisnioski+12','Mai+24','Krumholz+18_FT',
                         'Krumholz+18_Fonly','Ostriker+22_Fonly']
        
        # calculate vdisp from SigmaSFR_map
        #vdisp_map = np.zeros_like(SigmaSFR_map)
        
        # from Mai+24
        # log vdisp = 0.26836382*logSigmaSFR + 2.03763428
        
        # from Wisnioski+12
        # log vdisp = 0.61 * logSigmaSFR + 2.01
        
        # assum the minimum logSigmaSFR is -3.5, any smaller value give a 
        # constant vdisp
        
        # for spaxel that have very low SigmaSFR, give a certain value to 
        # avoid log10 (0)
        
        
        # for mavis, pix=0.02arcsec, the minimuum SigmaSFR using default prior is 3.4e-7
        #low_SigmaSFR = SigmaSFR_map < 1e-8
        #SigmaSFR_map[low_SigmaSFR] = 1e-8
        
        
        
        if paper == 'Wisnioski+12':
            log10_vdisp = 0.61*np.log10(SigmaSFR) + 2.01
            
            vdisp = np.power(10, log10_vdisp)
            
            
            if SigmaSFR<np.power(10,-1.5):
                vdisp = np.power(10, 0.61*-1.5 + 2.01)
        
        
            
            
        elif paper == 'Mai+24':
        
            log10_vdisp = 0.26836382*np.log10(SigmaSFR) + 2.03763428
        
            vdisp = np.power(10, log10_vdisp)
            if SigmaSFR<np.power(10,-3.5):
                vdisp = np.power(10, 0.26836382*-3.5 + 2.03763428)
        
        elif paper == 'Krumholz+18_FT':
            
            # don't use for loop ....
            #map_shape = vdisp_map.shape
            #for i in range(map_shape[0]):
            #    for j in range(map_shape[1]):
            #        vdisp_map[i,j] = K18_F_and_T(SigmaSFR_map[i,j])
                    
            SigmaSFR_break = K18_F_and_T_quick(get_break_point=True)
            Fonly_sigma = K18_F_and_T_quick(get_Fonly_sigma=True)
            if SigmaSFR > SigmaSFR_break:
            
                vdisp = K18_F_and_T_quick(SigmaSFR=SigmaSFR)
            else:
                vdisp = Fonly_sigma
        
        elif paper == 'Krumholz+18_Fonly':
            vdisp = K18_F_only(SigmaSFR)
        elif paper == 'Ostriker+22_Fonly':
            vdisp = O22_F_only(SigmaSFR)
        else:
            raise ValueError(f"Invalid option: '{paper}'. Valid options are: {', '.join(valid_options)}")
        
        return vdisp
    
    
    
    
    
    def make_all_blob_cube(self,hsimcube=True,sigmaSigmaSFRpaper='Wisnioski+12'):
        
        all_blob_cube = np.zeros((self.nw,self.ny,self.nx))
        meta_list = []
        for idx, row in self.blobs_df.iterrows():
            
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
            
            
            cube_i, meta_i = self.make_one_blob_cube(blob_row=row,
                                    sigmaSigmaSFRpaper=sigmaSigmaSFRpaper,
                                    hsimcube=hsimcube)
            all_blob_cube += cube_i
            
            meta_i["row_index"] = idx
            meta_list.append(meta_i)
        
        blob_all_info = pd.DataFrame(meta_list)
            
        return all_blob_cube, blob_all_info
    
    def arcsec_to_kpc(self,rad_in_arcsec,z):
        from astropy.cosmology import LambdaCDM
        lcdm = LambdaCDM(70,0.3,0.7)
        distance = lcdm.angular_diameter_distance(z).value # angular diameter distance, Mpc/radian
        rad_in_kpc = rad_in_arcsec * distance * np.pi/(180*3600)*1000
        return rad_in_kpc
    
    def make_fits(self,savepath,hsimcube=True,
                  sigmaSigmaSFRpaper='Wisnioski+12',csv_path=None):
        
        datacube,blob_info_df = self.make_all_blob_cube(hsimcube=hsimcube,
                              sigmaSigmaSFRpaper=sigmaSigmaSFRpaper)
        
        
        # Create a primary HDU
        hdu = fits.PrimaryHDU(datacube)
        
        # Modify the header
        hdu.header['OBJECT'] = 'Example Data'
        
        # spatial info
        hdu.header['NAXIS1'] = self.nx
        hdu.header['NAXIS2'] = self.ny
        
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        
        hdu.header['CDELT1'] = self.dx
        hdu.header['CDELT2'] = self.dy
        
        hdu.header['CUNIT1'] = 'arcsec'
        hdu.header['CUNIT2'] = 'arcsec'
        
        # spectral info
        hdu.header['NAXIS3'] = self.nw
        hdu.header['CTYPE3'] = 'wavelength'
        
        # recover the redshifted wavelength
        hdu.header['CDELT3'] = self.dw * (1 + self.redshift)
        
        hdu.header['SPECRES'] = self.lsf_fwhm  #lsf fwhm in Angstrom
        
        hdu.header['CRPIX3'] = 1
        # maybe I shouldn't add this 1/2*dw here
        #hdu.header['CRVAL3'] = (self.w_min + 1/2 * self.dw) * (1 + self.redshift)
        hdu.header['CRVAL3'] = (self.w_min ) * (1 + self.redshift)
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
        if csv_path is not None:
            blob_info_df.to_csv(csv_path)
        