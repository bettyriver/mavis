#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 14:53:18 2025

@author: ymai0110
"""
from fit_cube import fit_cube, ha_flux_to_SigmaSFR
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg

def get_input_output_SigmaSFR_vdisp(data_cube,snr_cube,construct_cube,
                                    sigmaSigmaSFRpaper,
             lsf_fwhm_angstrom=1.825,dx_arcsec_input=0.02,dx_arcsec_output=0.05):
    '''
    

    Parameters
    ----------
    data_cube : astropy.fits
        datacube fits file from MSIM output
    snr_cube : astropy.fits
        SNR cube fits file from MSIM outpu
    construct_cube : mavis.Construct_cube
        the Construct_cube for this file
    sigmaSigmaSFRpaper : str
        the paper from which we get sigma SigmaSFR relation
    lsf_fwhm_angstrom : float, optional
        the FWHM of LSF from MSIM output. The default is 1.825.
    dx_arcsec: float.
        the pixel size in arcsec for the MSIM output, the default is 0.05 (MAVIS 50 mas)

    Returns
    -------
    mfit_SigmaSFR_map : array
        SigmaSFR measure from MSIM output, in the unit of M_sun yr^-2 kpc^-2
    mfit_vdisp_valid :  array
        vdisp measure from MSIM output, in the unit of km/s
    SigmaSFR_map_input_big : array
        SigmaSFR input
    vdisp_map_input_big : array
        vdisp input

    '''
    gaussian_fit = fit_cube(data_cube=data_cube,
                            snr_cube=snr_cube,
                            lsf_fwhm_angstrom=lsf_fwhm_angstrom,
                            z=construct_cube.redshift,
                            savepath=None,snr_lim=4,dx_arcsec=dx_arcsec_output)
    mfit_flux = gaussian_fit[1].data
    mfit_vdisp = gaussian_fit[3].data
    
    
    query = mfit_vdisp>=0
    # plot vdisp hist
    mfit_vdisp_positive = mfit_vdisp[query]
    plt.hist(mfit_vdisp_positive,bins=30)
    plt.axvline(x=np.nanmedian(mfit_vdisp_positive),c='orange')
    ax = plt.gca()
    plt.text(0.7,0.9,
             r'output $\sigma$={:.2f}$\pm${:.2f}'.format(np.nanmedian(mfit_vdisp_positive),
                    np.nanstd(mfit_vdisp_positive)),
             color='orange',transform=ax.transAxes)
    plt.xlabel('velocity dispersion [km/s]')
    plt.ylabel('pixel number')
    #plt.savefig(parent_path+'plots/SFRrelatedsigma_velocity_dispersion_wisnioski12_msim.png',dpi=300,bbox_inches='tight')
    plt.show()
    
    
    flux_vdisp_corr = pg.corr(np.log10(mfit_flux[query]),np.log10(mfit_vdisp[query]))
    
    mfit_flux_valid = mfit_flux[query]
    mfit_vdisp_valid = mfit_vdisp[query]
    
    
    mfit_SigmaSFR_map = ha_flux_to_SigmaSFR(ha_flux_map=mfit_flux_valid,
                                       z=construct_cube.redshift,
                                       dx_arcsec=dx_arcsec_output,
                                       dy_arcsec=dx_arcsec_output)
    
    SigmaSFR_map_input = construct_cube.make_SigmaSFR_map()
    low_SigmaSFR = SigmaSFR_map_input<1e-7
    SigmaSFR_map_input[low_SigmaSFR] = 1e-7
    vdisp_map_input = construct_cube.make_vdisp_map(SFR_sigma=True,
                                sigmaSigmaSFRpaper=sigmaSigmaSFRpaper)
    query_input = np.log10(SigmaSFR_map_input)>-2
    
    # get array that SigmaSFR greater than certain value
    SigmaSFR_map_input_big = SigmaSFR_map_input[query_input]
    vdisp_map_input_big = vdisp_map_input[query_input]

    plt.scatter(np.log10(SigmaSFR_map_input_big),
                np.log10(vdisp_map_input_big),label='input')
    
    
    plt.scatter(np.log10(mfit_SigmaSFR_map),np.log10(mfit_vdisp_valid),
                label='MSIM fit (output)')
    
    plt.xlabel(r'log$_{10}$ $\Sigma_\mathrm{SFR}$ [$M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}$]')
    plt.ylabel(r'log$_{10}$ $\sigma_\mathrm{gas}$ [km/s]')
    #plt.yscale('log')
    ax = plt.gca()
    plt.text(0.7,0.1,r'output r={:.2f}'.format(flux_vdisp_corr['r'].values[0]),transform=ax.transAxes)
    plt.text(0.7,0.15,r'output p={:.2e}'.format(flux_vdisp_corr['p-val'].values[0]),transform=ax.transAxes)
    #plt.savefig(parent_path+'plots/flux_vs_velocity_dispersion_msim.png',dpi=300,bbox_inches='tight')
    plt.legend()
    plt.show()
    
    
    
    
    return mfit_SigmaSFR_map, mfit_vdisp_valid, SigmaSFR_map_input_big,vdisp_map_input_big
