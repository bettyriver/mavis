#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pre-process msim data for blobby3d

@author: Yifan Mai
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pandas
import sys 
sys.path.insert(0,'/Users/ymai0110/Documents/myPackages/metalpy/')
from meta import Metadata
from reconstructLSF import reconstruct_lsf
from gaussian_fit_mavis import psf_img_to_gauss, psf_img_to_one_gauss
from BPT import bptregion
import copy
import matplotlib as mpl
from matplotlib import colors
from numpy.ma import is_masked
from matplotlib.gridspec import GridSpec
import pandas as pd
import math

class PreBlobby3D:
    
    def __init__(
            self, flux_fits, variance_fits, redshift,save_path,emi_line='Ha_only'):
        
        self.wave_axis = 0
        
        self.redshift = redshift
        
        self.HD1 = flux_fits[0].header
        
        HD1 = self.HD1
        self.nx = HD1['NAXIS1']
        self.x_del = HD1['CDELT1']
        x_crpix = HD1['CRPIX1']
        self.ny = HD1['NAXIS2']
        self.y_del = HD1['CDELT2']
        y_crpix = HD1['CRPIX2']
        
        # convert the unit to 10^-20 erg/s/cm2/Ang (by times angular space)
        self.flux = flux_fits[0].data * self.x_del/1000 * self.y_del/1000 *1e-4 * 1e20
        self.var = variance_fits[0].data * self.x_del/1000 * self.y_del/1000 *1e-4 * 1e20
        self.cubeshape = self.flux.shape
        
        wavelength_crpix = HD1['CRPIX3']
        wavelength_crval = HD1['CRVAL3']*1e4
        self.wavelength_del_Wav = HD1['CDELT3']*1e4
        self.nw = HD1['NAXIS3']
        
        self.Wavelengths = wavelength_crval + (np.arange(0,self.nw)+1-wavelength_crpix)*self.wavelength_del_Wav
        self.Wavelengths_deredshift = self.Wavelengths/(1+self.redshift)
        
        # mas -> arcsec
        self.x_pix_range_sec = (np.arange(0,self.nx)+1-x_crpix)*self.x_del/1000
        self.x_pix_range = self.x_pix_range_sec - (self.x_pix_range_sec[0]+self.x_pix_range_sec[-1])/2

        self.y_pix_range_sec = (np.arange(0,self.ny)+1-y_crpix)*self.y_del/1000
        self.y_pix_range = self.y_pix_range_sec - (self.y_pix_range_sec[0]+self.y_pix_range_sec[-1])/2
    
        
        self.save_path = save_path
        self.emi_line = emi_line
    
    def cutout_data(self, snlim=None,xlim=None, ylim=None, wavelim=None):
        
        ''' 
        
        '''
        
        
        
        cutoutdata = self.cutout(data=self.flux,dim=3,xlim=xlim,ylim=ylim,wavelim=wavelim)
        cutoutvar = self.cutout(data=self.var,dim=3,xlim=xlim,ylim=ylim,wavelim=wavelim)
        
        
        
        wavelength_mask = np.ones(self.nw,dtype=bool)
        x_mask = np.full(self.nx,True)
        y_mask = np.full(self.ny,True)
        
        
        if wavelim is not None:
            wavelength_mask = (self.Wavelengths_deredshift > wavelim[0]) & (self.Wavelengths_deredshift < wavelim[1])
        if xlim is not None:
            x_values = np.arange(self.nx)
            x_mask = (x_values >= xlim[0]) & (x_values <= xlim[1])
        if ylim is not None:
            y_values = np.arange(self.ny)
            y_mask = (y_values >= ylim[0]) & (y_values <= ylim[1])
        
        
        cutout_nx = np.sum(x_mask)
        cutout_ny = np.sum(y_mask)
        cutout_nw = np.sum(wavelength_mask)
        
        if self.wave_axis == 0:
            data_for_save = cutoutdata.reshape(cutout_nw,cutout_nx*cutout_ny).T
            var_for_save = cutoutvar.reshape(cutout_nw,cutout_nx*cutout_ny).T
        if self.wave_axis == 2:
            data_for_save = cutoutdata.reshape(cutout_nx*cutout_ny,cutout_nw)
            var_for_save = cutoutvar.reshape(cutout_nx*cutout_ny,cutout_nw)
        
        
        
        

        metadata = np.array([cutout_ny,cutout_nx,cutout_nw,
                             self.x_pix_range[x_mask][0]-1/2*self.x_del/1000,self.x_pix_range[x_mask][-1]+1/2*self.x_del/1000,
                             self.y_pix_range[y_mask][0]-1/2*self.y_del/1000,self.y_pix_range[y_mask][-1]+1/2*self.y_del/1000,
                             self.Wavelengths_deredshift[wavelength_mask][0]-1/2*self.wavelength_del_Wav/(1+self.redshift),
                             self.Wavelengths_deredshift[wavelength_mask][-1]+1/2*self.wavelength_del_Wav/(1+self.redshift)])
        
        return data_for_save, var_for_save, metadata
        
    
    def cutout(self,data,dim, xlim=None, ylim=None, wavelim=None):
        x_mask = np.full(self.nx,True)
        y_mask = np.full(self.ny,True)
        if dim == 3:
            wavelength_mask = np.ones(self.nw,dtype=bool)
            
            
            
            if wavelim is not None:
                wavelength_mask = (self.Wavelengths_deredshift > wavelim[0]) & (self.Wavelengths_deredshift < wavelim[1])
            if xlim is not None:
                x_values = np.arange(self.nx)
                x_mask = (x_values >= xlim[0]) & (x_values <= xlim[1])
            if ylim is not None:
                y_values = np.arange(self.ny)
                y_mask = (y_values >= ylim[0]) & (y_values <= ylim[1])
            
            if self.wave_axis == 0:
                cutoutdata = data[wavelength_mask,:,:][:, y_mask, :][:, :, x_mask]
                
            # I think it's wrong... not need to think about it at this time...
            #if self.wave_axis == 2:
            #    cutoutdata = data[:, x_mask, :][:, :, y_mask][wavelength_mask,:,:]
        if dim == 2:
            if xlim is not None:
                x_values = np.arange(self.nx)
                x_mask = (x_values >= xlim[0]) & (x_values <= xlim[1])
            if ylim is not None:
                y_values = np.arange(self.ny)
                y_mask = (y_values >= ylim[0]) & (y_values <= ylim[1])
            
            cutoutdata = data[ y_mask, :][ :, x_mask]
        
        
        return cutoutdata
    
    
    def model_options(self,inc,psf_fits,lsf_fwhm_ang,flat_vdisp=True,psfimg=True,gaussian=2,
                      constrain_kinematics=None,constrain_vdisp=False,
                      constrain_pa=None,vsys_set=None,
                      mavis_test=False):
        
        
        '''
        inc: float
            inclination in rad
        
        
        emi_line: Ha or Oii
            constrain_kinematics: path of kinematic parameter csv file
        
        constrain_kinematics: constrain vel profile, pa
        constrain_vdisp: when constrain_kinematics is not None, constrain
                         vel profile, pa, vdisp
        constrain_pa: constrain pa only
        mavis_test: the model options set for mavis science project
        lsf_fwhm_ang: LSF FWHM in angstrom, in observe frame
        
        '''
        
        if self.emi_line =='Ha' or self.emi_line=='Ha_only':
            emi_line_wave = 6563
            
        elif self.emi_line == 'Oii':
            # oii 3727 and oii 3729, choose a wavelength in between
            emi_line_wave = 3728
            
        elif self.emi_line == 'Hb':
            # Hbeta line wavelength in vacuum 4862.683 
            emi_line_wave = 4862.683
            
        
        
        
        
        
        modelfile = open(self.save_path+"MODEL_OPTIONS","w")
        # first line
        modelfile.write('# Specify model options\n')
        
        
        # LSFFWHM
        # reconstruct return sigma
        
        lsf_fwhm = lsf_fwhm_ang/(1+self.redshift)
        
        
        modelfile.write('LSFFWHM\t%.4f\n'%(lsf_fwhm))
        
        if psfimg==True:
            # default z-band
            img = psf_fits[0].data
            
            psf_header = psf_fits[0].header
            pix_size_arcsec = psf_header['CDELT1']/1000 # CDELT1 in unit of mas
            
            
            if gaussian==2:
                weight1, weight2, fwhm1, fwhm2 = psf_img_to_gauss(img,
                                                    pix_size_arcsec=pix_size_arcsec)
                
                modelfile.write('PSFWEIGHT\t%f %f\n'%(weight1,weight2))
                modelfile.write('PSFFWHM\t%f %f\n'%(fwhm1,fwhm2))
                
            elif gaussian==1:
                
                weight1, fwhm1 = psf_img_to_one_gauss(img,
                                                    pix_size_arcsec=pix_size_arcsec)
                
                modelfile.write('PSFWEIGHT\t%f\n'%(weight1))
                modelfile.write('PSFFWHM\t%f\n'%(fwhm1))
                
                
                
                
            else:
                print('error... Gaussian component num no supported')
            #if gaussian==3:
            #    weight1, weight2, weight3, fwhm1, fwhm2 ,fwhm3= mofgauFit.psf_img_to_gauss_three(img)
            #    
            #    modelfile.write('PSFWEIGHT\t%f %f %f\n'%(weight1,weight2,weight3))
            #    modelfile.write('PSFFWHM\t%f %f %f\n'%(fwhm1,fwhm2,fwhm3))
                
                
                
                
        else:
            print('error... PSF fitting method not supported')
            #PSF
            #psfhdr = self.fitsdata[4].header
            # note that the datacubes have mistake, alpha is beta , beta is alpha
            #beta = psfhdr['MAGPI PSF ZBAND MOFFAT ALPHA']
            #alpha = psfhdr['MAGPI PSF ZBAND MOFFAT BETA']
            
            #weight1, weight2, fwhm1, fwhm2 = mofgauFit.mof_to_gauss(alpha=alpha, 
            #                                                        beta=beta)
            #modelfile.write('PSFWEIGHT\t%f %f\n'%(weight1,weight2))
            #modelfile.write('PSFFWHM\t%f %f\n'%(fwhm1,fwhm2))
        
        # inclination
        
        
        modelfile.write('INC\t%f\n'%(inc))
        if self.emi_line =='Ha':
            modelfile.write('LINE\t6562.81\n')
            modelfile.write('LINE\t6583.1\t6548.1\t0.333\n')
        elif self.emi_line=='Ha_only':
            modelfile.write('LINE\t6562.81\n')
        elif self.emi_line == 'Oii':
            modelfile.write('LINE\t3727.092\n')
            modelfile.write('LINE\t3729.875\n')
        elif self.emi_line == 'Hb':
            modelfile.write('LINE\t4862.683\n')
        
        
        if flat_vdisp == True:
            modelfile.write('VDISPN_SIGMA\t1.000000e-09\n')
        
        if constrain_pa is not None:
            kinematics_df = pd.read_csv(constrain_pa)
            
            pa_array = kinematics_df['PA'].to_numpy()
            pa_mean, pa_std = self.circular_mean_and_std(pa_array)
            
            if pa_std>0.000001: # if too small, then pa_min would = pa_max when round to 6 decimal places
                modelfile.write('PA_MIN\t{:.6f}\n'.format(pa_mean - pa_std))
                modelfile.write('PA_MAX\t{:.6f}\n'.format(pa_mean + pa_std))
            else:
                modelfile.write('PA_MIN\t{:.6f}\n'.format(pa_mean - np.power(0.1,5-self.get_order(pa_mean))))
                modelfile.write('PA_MAX\t{:.6f}\n'.format(pa_mean + np.power(0.1,5-self.get_order(pa_mean))))
            
        
        if constrain_kinematics is not None:
            kinematics_df = pd.read_csv(constrain_kinematics)
            # vsys to be free
            i = 0
            vc = kinematics_df['VMAX'][i]
            rt = kinematics_df['VSLOPE'][i]
            beta = kinematics_df['VBETA'][i]
            gamma = kinematics_df['VGAMMA'][i]
            pa_array = kinematics_df['PA'].to_numpy()
            #vsys = kinematics_df['VSYS'][i]
            
            modelfile.write('VC_MIN\t{:.6f}\n'.format(vc - np.power(0.1,5-self.get_order(vc))))
            modelfile.write('VC_MAX\t{:.6f}\n'.format(vc + np.power(0.1,5-self.get_order(vc))))
            
            modelfile.write('VSLOPE_MIN\t{:.6f}\n'.format(rt - np.power(0.1,5-self.get_order(rt))))
            modelfile.write('VSLOPE_MAX\t{:.6f}\n'.format(rt + np.power(0.1,5-self.get_order(rt))))
            
            modelfile.write('VBETA_MIN\t{:.6f}\n'.format(beta - np.power(0.1,5-self.get_order(beta))))
            modelfile.write('VBETA_MAX\t{:.6f}\n'.format(beta + np.power(0.1,5-self.get_order(beta))))
            
            modelfile.write('VGAMMA_MIN\t{:.6f}\n'.format(gamma - np.power(0.1,5-self.get_order(gamma))))
            modelfile.write('VGAMMA_MAX\t{:.6f}\n'.format(gamma + np.power(0.1,5-self.get_order(gamma))))
            
            pa_mean, pa_std = self.circular_mean_and_std(pa_array)
            
            if pa_std>0.000001:
                modelfile.write('PA_MIN\t{:.6f}\n'.format(pa_mean - pa_std))
                modelfile.write('PA_MAX\t{:.6f}\n'.format(pa_mean + pa_std))
            else:
                modelfile.write('PA_MIN\t{:.6f}\n'.format(pa_mean - np.power(0.1,5-self.get_order(pa_mean))))
                modelfile.write('PA_MAX\t{:.6f}\n'.format(pa_mean + np.power(0.1,5-self.get_order(pa_mean))))
            
            
            if constrain_vdisp == True:
                # vdisp take median value of all effective samples
                vdisp = kinematics_df['VDISP0'].median()
                modelfile.write('LOGVDISP0_MIN\t{:.6f}\n'.format(vdisp - np.power(0.1,5-self.get_order(vdisp))))
                modelfile.write('LOGVDISP0_MAX\t{:.6f}\n'.format(vdisp + np.power(0.1,5-self.get_order(vdisp))))
        if vsys_set is not None:
            modelfile.write('VSYS_MAX\t{:.1f}\n'.format(vsys_set))
        
        
        if mavis_test:
            # set the number of blobs fixed to 300
            modelfile.write('NMAX\t300\n') 
            # 1 is true, 0 is false
            modelfile.write('NFIXED\t1\n') 
            # 0.02 arcsec, 1/10 of pixel
            modelfile.write('RADIUSLIM_MIN\t0.02\n') 
            # 5 arcsec, just a test, need to try different values...
            modelfile.write('RADIUSLIM_MAX\t5\n') 
            
        
        modelfile.close()
        
    def savedata(self,data,var,metadata):
        

        metafile = open(self.save_path+"metadata.txt","w")
        metafile.write("%d %d %d %.3f %.3f %.3f %.3f %.3f %.3f"%(metadata[0],metadata[1],
                                                                 metadata[2],metadata[3],
                                                                 metadata[4],metadata[5],
                                                                 metadata[6],metadata[7],
                                                                 metadata[8]))

        metafile.close()

        np.savetxt(self.save_path+"data.txt",np.nan_to_num(data))
        np.savetxt(self.save_path+"var.txt",np.nan_to_num(var))
    
    def dn4_options(self,iterations=5000,new_level_interval=10000,
                    save_interval=10000,number_of_particles=1):
        modelfile = open(self.save_path+"OPTIONS","w")
        
        
        modelfile.write('# File containing parameters for DNest4\n')
        modelfile.write('# Put comments at the top, or at the end of the line.\n')
        modelfile.write('%d	# Number of particles\n'%(number_of_particles))
        modelfile.write('%d	# New level interval\n'%(new_level_interval))
        modelfile.write('%d	# Save interval\n'%(save_interval))
        modelfile.write('100	# Thread steps - how many steps each thread should do independently before communication\n')
        modelfile.write('0	# Maximum number of levels\n')
        modelfile.write('10	# Backtracking scale length (lambda in the paper)\n')
        modelfile.write('100	# Strength of effect to force histogram to equal push (beta in the paper)\n')
        modelfile.write('%d	# Maximum number of saves (0 = infinite)\n'%(iterations))
        modelfile.write('sample.txt	# Sample file\n')
        modelfile.write('sample_info.txt	# Sample info file\n')
        modelfile.write('levels.txt	# Sample file\n')
        modelfile.close()
    def get_order(self,num):
        # Handling zero separately because log10(0) is undefined
        if num == 0:
            raise ValueError("log10(0) is undefined, so the order of 0 is not defined.")
        
        # Calculate the order using log10 and floor function
        order = math.floor(math.log10(abs(num)))
        
        if order<0:
            order = -1
        
        return order
    
    def circular_mean_and_std(self,angles_rad):
        x = np.cos(angles_rad)
        y = np.sin(angles_rad)
        
        # Compute the mean direction
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # Mean angle in radians
        mean_angle_rad = np.arctan2(mean_y, mean_x)
        
        # the above func return [-pi,pi], but I prefer[0,2pi], so plus
        # 2pi for negative angles.....
        if mean_angle_rad<0:
            mean_angle_rad = mean_angle_rad + 2*np.pi
        
        # Resultant vector length
        R = np.sqrt(mean_x**2 + mean_y**2)
        
        # Circular standard deviation in radians
        circular_std_rad = np.sqrt(-2 * np.log(R))
        
        return mean_angle_rad, circular_std_rad