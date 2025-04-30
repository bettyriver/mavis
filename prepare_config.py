#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 16:09:43 2025

@author: ymai0110
"""

import numpy as np

def write_lsf_data(savepath,lsf_fwhm,wavelength_range):
    '''
    write the LSF data document. This function is only for LSF is a constant
    as a function of wavelength

    Parameters
    ----------
    savepath : str
        path to save the document.
    lsf_fwhm : float
        FWHM of LSF in Angstrom.
    wavelength_range : tuple
        (wave_min, wave_max).

    Returns
    -------
    None.

    '''
    file = open(savepath,'w')
    file.write('# Column 1: Wavelength vector in Angstrom\n')
    file.write('# Column 2: FWHM of observation at the '
               'respective wavelength in Angstrom\n')
    file.write('# Spectral resolution LSF FWHM ' + str(lsf_fwhm) +
               ' Angstrom\n')
    file.write('# \n')
    wave_min = wavelength_range[0]
    wave_max = wavelength_range[1]
    
    wave_arr = np.linspace(wave_min, wave_max, 
                           num=int((wave_max-wave_min)/lsf_fwhm*4))
    
    for wave in wave_arr:
        file.write('{:.4f}\t{:.4f}\n'.format(wave,lsf_fwhm))
    
    file.close()