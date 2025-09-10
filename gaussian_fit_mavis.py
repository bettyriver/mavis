#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 13:57:05 2025

@author: ymai0110
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import astropy.io.fits as fits
import matplotlib as mpl

def moffat(x,y,a,b):
    f = (b-1)/np.pi/a/a*np.power((1+(x**2+y**2)/a**2),-b)
    return f

def two_gaussian(xdata_tuple,A1,A2,sigma1,sigma2):
    (x, y) = xdata_tuple
    f = A1*np.exp(-(x**2+y**2)/2/sigma1**2)+A2*np.exp(-(x**2+y**2)/2/sigma2**2)
    return f.ravel()

def one_gaussian_c(xdata_tuple,A1,sigma1,x0,y0):
    '''
    one gaussian, center flexible
    
    
    '''
    
    (x, y) = xdata_tuple
    f = A1*np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma1**2)
    
    return f.ravel()
    
    

def two_gaussian_c(xdata_tuple,A1,A2,sigma1,sigma2,x0,y0):
    '''
    
    shift center compare to the two_gaussian
    
    Parameters
    ----------
    xdata_tuple : TYPE
        DESCRIPTION.
    A1 : TYPE
        DESCRIPTION.
    A2 : TYPE
        DESCRIPTION.
    sigma1 : TYPE
        DESCRIPTION.
    sigma2 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    (x, y) = xdata_tuple
    f = A1*np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma1**2)+A2*np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma2**2)
    return f.ravel()

def three_gaussian(xdata_tuple,A1,A2,A3,sigma1,sigma2,sigma3):
    (x, y) = xdata_tuple
    f = A1*np.exp(-(x**2+y**2)/2/sigma1**2)+A2*np.exp(-(x**2+y**2)/2/sigma2**2)+A3*np.exp(-(x**2+y**2)/2/sigma3**2)
    return f.ravel()

def three_gaussian_c(xdata_tuple,A1,A2,A3,sigma1,sigma2,sigma3,x0,y0):
    (x, y) = xdata_tuple
    f = A1*np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma1**2)+A2*np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma2**2)+A3*np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma3**2)
    return f.ravel()



def psf_img_to_one_gauss(img,pix_size_arcsec,plot=False):
    '''
    img: psf image
    ---
    psf img to one Gaussian fitting 
    
    
    
    '''
    
    y = np.arange(img.shape[0]) - img.shape[0]/2
    x = np.arange(img.shape[1]) - img.shape[1]/2
    
    xx, yy = np.meshgrid(x, y)
    
    xdata = np.vstack((xx.ravel(),yy.ravel()))
    ydata = img.ravel()
    
    popt, pcov = curve_fit(f=one_gaussian_c,xdata=xdata,ydata=ydata,
                           bounds=([0,0.1,-5,-5],[np.inf,np.inf,5,5]))
    
    t2 = one_gaussian_c(xdata,popt[0],popt[1],popt[2],popt[3]).reshape(img.shape[0],img.shape[1])
    
    if plot==True:
        
        fig,ax = plt.subplots(1,5,figsize=(18,8))
        
        max_v = np.max(img)
        min_v = np.min(img)
        
        im0=ax[0].imshow(img,vmin=min_v, vmax=max_v)
        cb0 = plt.colorbar(im0,ax=ax[0],fraction=0.047)
        cb0.ax.locator_params(nbins=5)
        ax[0].set_title('psf img')
        
        
        
        im1=ax[1].imshow(t2,vmin=min_v, vmax=max_v)
        cb1 = plt.colorbar(im1,ax=ax[1],fraction=0.047)
        cb1.ax.locator_params(nbins=5)
        ax[1].set_title('one_gaussian')
        
        max_res = np.max((img-t2)/t2)
        
        im2=ax[2].imshow((img-t2),cmap='RdYlBu',vmin=-0.002,vmax=0.002)
        cb2 = plt.colorbar(im2,ax=ax[2],fraction=0.047)
        cb2.ax.locator_params(nbins=5)
        ax[2].set_title('residual')
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.8, 
                    hspace=0.4)
        
        
        
        ax[3].plot(x,img[:,int(img.shape[0]/2+0.5)],label='img')
        ax[3].plot(x,t2[:,int(img.shape[0]/2+0.5)],label='fit')
        ax[3].legend()
        
        ax[4].plot(x,img[:,int(img.shape[0]/2+0.5)],label='img')
        ax[4].plot(x,t2[:,int(img.shape[0]/2+0.5)],label='fit')
        ax[4].set_ylim(0,0.002)
        ax[4].legend()
        
        
        plt.show()
    
    
    max_flux = np.max(img)
    rr = np.sqrt(xx**2 + yy**2)
    fwhm = 2*np.min(rr[img < max_flux/2])
    print(fwhm)
    print(popt)
    
    # wrong !! shouldn't be amplitude of Gaussian 
    #weight1 = popt[0]/(popt[0]+popt[1])
    #weight2 = popt[1]/(popt[0]+popt[1])
    
    # should be integral of Gaussian over the entire xy plane
    # integral = A*2*pi*sigma^2
    
    
    # /* don't need to calculate weight for one gaussian... */
    #integral_1 = popt[0]*2*np.pi*(popt[2]**2)
    #integral_2 = popt[1]*2*np.pi*(popt[3]**2)
    #weight1 = integral_1/(integral_1+integral_2)
    #weight2 = integral_2/(integral_1+integral_2)
    
    
    # times arcsec per pixel
    gau_fwhm1 = 2*np.sqrt(2*np.log(2))*np.abs(popt[1])*pix_size_arcsec
    
    if (popt[0] < 0) :
        print('something went wrong!')
        return 0
    
    # weight is 1 for one Gaussian
    return 1, gau_fwhm1
    
    



def psf_img_to_gauss(img,pix_size_arcsec,plot=False):
    
    '''
    img: psf image
    ---
    psf img to two Gaussian fitting (Default)
    
    '''
    y = np.arange(img.shape[0]) - img.shape[0]/2
    x = np.arange(img.shape[1]) - img.shape[1]/2
    
    xx, yy = np.meshgrid(x, y)
    
    xdata = np.vstack((xx.ravel(),yy.ravel()))
    ydata = img.ravel()
    
    popt, pcov = curve_fit(f=two_gaussian_c,xdata=xdata,ydata=ydata,
                           bounds=([0,0,0.1,0.1,-5,-5],[np.inf,np.inf,np.inf,np.inf,5,5]))
    
    t2 = two_gaussian_c(xdata,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]).reshape(img.shape[0],img.shape[1])
    
    if plot==True:
        
        fig,ax = plt.subplots(1,5,figsize=(18,8))
        
        max_v = np.max(img)
        min_v = np.min(img)
        
        im0=ax[0].imshow(img,vmin=min_v, vmax=max_v)
        cb0 = plt.colorbar(im0,ax=ax[0],fraction=0.047)
        cb0.ax.locator_params(nbins=5)
        ax[0].set_title('psf img')
        
        
        
        im1=ax[1].imshow(t2,vmin=min_v, vmax=max_v)
        cb1 = plt.colorbar(im1,ax=ax[1],fraction=0.047)
        cb1.ax.locator_params(nbins=5)
        ax[1].set_title('two_gaussian')
        
        max_res = np.max((img-t2)/t2)
        
        im2=ax[2].imshow((img-t2),cmap='RdYlBu',vmin=-0.002,vmax=0.002)
        cb2 = plt.colorbar(im2,ax=ax[2],fraction=0.047)
        cb2.ax.locator_params(nbins=5)
        ax[2].set_title('residual')
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.8, 
                    hspace=0.4)
        
        
        
        ax[3].plot(x,img[:,int(img.shape[0]/2+0.5)],label='img')
        ax[3].plot(x,t2[:,int(img.shape[0]/2+0.5)],label='fit')
        ax[3].legend()
        
        ax[4].plot(x,img[:,int(img.shape[0]/2+0.5)],label='img')
        ax[4].plot(x,t2[:,int(img.shape[0]/2+0.5)],label='fit')
        ax[4].set_ylim(0,0.002)
        ax[4].legend()
        
        
        plt.show()
    
    
    max_flux = np.max(img)
    rr = np.sqrt(xx**2 + yy**2)
    fwhm = 2*np.min(rr[img < max_flux/2])
    print(fwhm)
    print(popt)
    
    # wrong !! shouldn't be amplitude of Gaussian 
    #weight1 = popt[0]/(popt[0]+popt[1])
    #weight2 = popt[1]/(popt[0]+popt[1])
    
    # should be integral of Gaussian over the entire xy plane
    # integral = A*2*pi*sigma^2
    integral_1 = popt[0]*2*np.pi*(popt[2]**2)
    integral_2 = popt[1]*2*np.pi*(popt[3]**2)
    weight1 = integral_1/(integral_1+integral_2)
    weight2 = integral_2/(integral_1+integral_2)
    
    
    # times arcsec per pixel
    gau_fwhm1 = 2*np.sqrt(2*np.log(2))*np.abs(popt[2])*pix_size_arcsec
    gau_fwhm2 = 2*np.sqrt(2*np.log(2))*np.abs(popt[3])*pix_size_arcsec
    
    if (popt[0] < 0) | (popt[1] < 0):
        print('something went wrong!')
        return 0
    
    return weight1, weight2, gau_fwhm1, gau_fwhm2

def mof_to_gauss(alpha,beta,pix_size_arcsec,plot=False):

    #beta = 42.77066906841987
    #alpha = 13.38382023870645
    
    x = np.linspace(-24, 24, 1000)
    y = np.linspace(-24, 24, 1000)
    xx, yy = np.meshgrid(x, y)
    
    #t0 = moffat(x, y, alpha, beta)
    t1 = moffat(xx,yy,alpha,beta)
    
    
    
    xdata = np.vstack((xx.ravel(),yy.ravel()))
    ydata = t1.ravel()
    
    popt, pcov = curve_fit(f=two_gaussian,xdata=xdata,ydata=ydata,
                           bounds=([0,0,0,0],[np.inf,np.inf,np.inf,np.inf]))
    
    t2 = two_gaussian(xdata,popt[0],popt[1],popt[2],popt[3]).reshape(1000,1000)
    
    if plot==True:
        
        fig,ax = plt.subplots(1,3)
        
        max_v = np.max(t1)
        min_v = np.min(t1)
        
        im0=ax[0].imshow(t1,vmin=min_v, vmax=max_v)
        cb0 = plt.colorbar(im0,ax=ax[0],fraction=0.047)
        cb0.ax.locator_params(nbins=5)
        ax[0].set_title('moffat')
        
        
        
        im1=ax[1].imshow(t2,vmin=min_v, vmax=max_v)
        cb1 = plt.colorbar(im1,ax=ax[1],fraction=0.047)
        cb1.ax.locator_params(nbins=5)
        ax[1].set_title('two_gaussian')
        
        max_res = np.max(t1-t2)
        
        im2=ax[2].imshow(t1-t2,cmap='RdYlBu',vmin=-max_res, vmax=max_res)
        cb2 = plt.colorbar(im2,ax=ax[2],fraction=0.047)
        cb2.ax.locator_params(nbins=5)
        ax[2].set_title('residual')
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.8, 
                    hspace=0.4)
        
        plt.show()
    
    
    max_flux = np.max(t1)
    rr = np.sqrt(xx**2 + yy**2)
    fwhm = 2*np.min(rr[t1 < max_flux/2])
    print(fwhm)
    print(popt)
    
    weight1 = popt[0]/(popt[0]+popt[1])
    weight2 = popt[1]/(popt[0]+popt[1])
    
    # times arcsec per pixel
    gau_fwhm1 = 2*np.sqrt(2*np.log(2))*np.abs(popt[2])*pix_size_arcsec
    gau_fwhm2 = 2*np.sqrt(2*np.log(2))*np.abs(popt[3])*pix_size_arcsec
    
    if (popt[0] < 0) | (popt[1] < 0):
        print('something went wrong!')
        return 0
    
    return weight1, weight2, gau_fwhm1, gau_fwhm2

def psf_img_to_gauss_three(img,pix_size_arcsec,plot=False):
    
    '''
    img: psf image
    
    
    '''
    y = np.arange(img.shape[0]) - img.shape[0]/2
    x = np.arange(img.shape[1]) - img.shape[1]/2
    
    xx, yy = np.meshgrid(x, y)
    
    xdata = np.vstack((xx.ravel(),yy.ravel()))
    ydata = img.ravel()
    
    popt, pcov = curve_fit(f=three_gaussian_c,xdata=xdata,ydata=ydata,
                           bounds=([0,0,0,0.1,0.1,0.1,-5,-5],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,5,5]))
    
    t2 = three_gaussian_c(xdata,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7]).reshape(img.shape[0],img.shape[1])
    
    if plot==True:
        
        fig,ax = plt.subplots(1,5,figsize=(18,8))
        
        max_v = np.max(img)
        min_v = np.min(img)
        
        im0=ax[0].imshow(img,vmin=min_v, vmax=max_v)
        cb0 = plt.colorbar(im0,ax=ax[0],fraction=0.047)
        cb0.ax.locator_params(nbins=5)
        ax[0].set_title('psf img')
        
        
        
        im1=ax[1].imshow(t2,vmin=min_v, vmax=max_v)
        cb1 = plt.colorbar(im1,ax=ax[1],fraction=0.047)
        cb1.ax.locator_params(nbins=5)
        ax[1].set_title('three_gaussian')
        
        max_res = np.max((img-t2)/t2)
        
        im2=ax[2].imshow((img-t2)/t2,cmap='RdYlBu',vmin=-0.1,vmax=0.1)
        cb2 = plt.colorbar(im2,ax=ax[2],fraction=0.047)
        cb2.ax.locator_params(nbins=5)
        ax[2].set_title('residual')
        
        
        ax[3].plot(x,img[:,int(img.shape[0]/2+0.5)],label='img')
        ax[3].plot(x,t2[:,int(img.shape[0]/2+0.5)],label='fit')
        ax[3].legend()
        
        ax[4].plot(x,img[:,int(img.shape[0]/2+0.5)],label='img')
        ax[4].plot(x,t2[:,int(img.shape[0]/2+0.5)],label='fit')
        ax[4].set_ylim(0,0.002)
        ax[4].legend()
        
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.8, 
                    hspace=0.4)
        
        plt.show()
    
    
    max_flux = np.max(img)
    rr = np.sqrt(xx**2 + yy**2)
    fwhm = 2*np.min(rr[img < max_flux/2])
    print(fwhm)
    print(popt)
    
    weight1 = popt[0]/(popt[0]+popt[1]+popt[2])
    weight2 = popt[1]/(popt[0]+popt[1]+popt[2])
    weight3 = popt[2]/(popt[0]+popt[1]+popt[2])
    
    # 0.2 arcsec per pixel
    gau_fwhm1 = 2*np.sqrt(2*np.log(2))*np.abs(popt[3])*pix_size_arcsec
    gau_fwhm2 = 2*np.sqrt(2*np.log(2))*np.abs(popt[4])*pix_size_arcsec
    gau_fwhm3 = 2*np.sqrt(2*np.log(2))*np.abs(popt[5])*pix_size_arcsec
    
    
    if (popt[0] < 0) | (popt[1] < 0):
        print('something went wrong!')
        return 0
    
    return weight1, weight2, weight3, gau_fwhm1, gau_fwhm2, gau_fwhm3

def mof_to_gauss(alpha,beta,pix_size_arcsec,plot=False):

    #beta = 42.77066906841987
    #alpha = 13.38382023870645
    
    x = np.linspace(-24, 24, 1000)
    y = np.linspace(-24, 24, 1000)
    xx, yy = np.meshgrid(x, y)
    
    #t0 = moffat(x, y, alpha, beta)
    t1 = moffat(xx,yy,alpha,beta)
    
    
    
    xdata = np.vstack((xx.ravel(),yy.ravel()))
    ydata = t1.ravel()
    
    popt, pcov = curve_fit(f=two_gaussian,xdata=xdata,ydata=ydata,
                           bounds=([0,0,0.1,0.1],[np.inf,np.inf,np.inf,np.inf]))
    
    t2 = two_gaussian(xdata,popt[0],popt[1],popt[2],popt[3]).reshape(1000,1000)
    
    if plot==True:
        
        fig,ax = plt.subplots(1,3)
        
        max_v = np.max(t1)
        min_v = np.min(t1)
        
        im0=ax[0].imshow(t1,vmin=min_v, vmax=max_v)
        cb0 = plt.colorbar(im0,ax=ax[0],fraction=0.047)
        cb0.ax.locator_params(nbins=5)
        ax[0].set_title('moffat')
        
        
        
        im1=ax[1].imshow(t2,vmin=min_v, vmax=max_v)
        cb1 = plt.colorbar(im1,ax=ax[1],fraction=0.047)
        cb1.ax.locator_params(nbins=5)
        ax[1].set_title('two_gaussian')
        
        max_res = np.max(t1-t2)
        
        im2=ax[2].imshow(t1-t2,cmap='RdYlBu',vmin=-max_res, vmax=max_res)
        cb2 = plt.colorbar(im2,ax=ax[2],fraction=0.047)
        cb2.ax.locator_params(nbins=5)
        ax[2].set_title('residual')
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.8, 
                    hspace=0.4)
        
        plt.show()
    
    
    max_flux = np.max(t1)
    rr = np.sqrt(xx**2 + yy**2)
    fwhm = 2*np.min(rr[t1 < max_flux/2])
    print(fwhm)
    print(popt)
    
    weight1 = popt[0]/(popt[0]+popt[1])
    weight2 = popt[1]/(popt[0]+popt[1])
    
    # multiply arcsec per pixel
    gau_fwhm1 = 2*np.sqrt(2*np.log(2))*np.abs(popt[2])*pix_size_arcsec
    gau_fwhm2 = 2*np.sqrt(2*np.log(2))*np.abs(popt[3])*pix_size_arcsec
    
    if (popt[0] < 0) | (popt[1] < 0):
        print('something went wrong!')
        return 0
    
    return weight1, weight2, gau_fwhm1, gau_fwhm2
    
def mof_to_gauss_threeGau(alpha,beta,pix_size_arcsec,plot=False):

    #beta = 42.77066906841987
    #alpha = 13.38382023870645
    
    x = np.linspace(-24, 24, 1000)
    y = np.linspace(-24, 24, 1000)
    xx, yy = np.meshgrid(x, y)
    
    #t0 = moffat(x, y, alpha, beta)
    t1 = moffat(xx,yy,alpha,beta)
    
    
    
    xdata = np.vstack((xx.ravel(),yy.ravel()))
    ydata = t1.ravel()
    
    popt, pcov = curve_fit(f=three_gaussian,xdata=xdata,ydata=ydata,
                           bounds=([0,0,0,0.1,0.1,0.1],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]))
    
    t2 = three_gaussian(xdata,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]).reshape(1000,1000)
    
    if plot==True:
        
        fig,ax = plt.subplots(1,3)
        
        max_v = np.max(t1)
        min_v = np.min(t1)
        
        im0=ax[0].imshow(t1,vmin=min_v, vmax=max_v)
        cb0 = plt.colorbar(im0,ax=ax[0],fraction=0.047)
        cb0.ax.locator_params(nbins=5)
        ax[0].set_title('moffat')
        
        
        
        im1=ax[1].imshow(t2,vmin=min_v, vmax=max_v)
        cb1 = plt.colorbar(im1,ax=ax[1],fraction=0.047)
        cb1.ax.locator_params(nbins=5)
        ax[1].set_title('three_gaussian')
        
        max_res = np.max(t1-t2)
        
        im2=ax[2].imshow(t1-t2,cmap='RdYlBu',vmin=-max_res, vmax=max_res)
        cb2 = plt.colorbar(im2,ax=ax[2],fraction=0.047)
        cb2.ax.locator_params(nbins=5)
        ax[2].set_title('residual')
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.8, 
                    hspace=0.4)
        
        plt.show()
    
    
    max_flux = np.max(t1)
    rr = np.sqrt(xx**2 + yy**2)
    fwhm = 2*np.min(rr[t1 < max_flux/2])
    print(fwhm)
    print(popt)
    
    # ! unfinished
    
    weight1 = popt[0]/(popt[0]+popt[1])
    weight2 = popt[1]/(popt[0]+popt[1])
    
    # 0.2 arcsec per pixel
    gau_fwhm1 = 2*np.sqrt(2*np.log(2))*np.abs(popt[2])*pix_size_arcsec
    gau_fwhm2 = 2*np.sqrt(2*np.log(2))*np.abs(popt[3])*pix_size_arcsec
    
    if (popt[0] < 0) | (popt[1] < 0):
        print('something went wrong!')
        return 0
    
    return weight1, weight2, gau_fwhm1, gau_fwhm2

def plot_compar(datapath,magpiid):
    magpifile = fits.open(datapath + "MAGPI"+str(magpiid)+"_minicube.fits")
    psfhdr = magpifile[4].header
    # note that the datacubes have mistake, alpha is beta , beta is alpha
    beta = psfhdr['MAGPI PSF ZBAND MOFFAT ALPHA']
    alpha = psfhdr['MAGPI PSF ZBAND MOFFAT BETA']
    
    weight1, weight2, fwhm1, fwhm2 = mof_to_gauss(alpha=alpha, beta=beta, plot=True,magpiid=magpiid)
    
def plot_compar_threeGau(datapath,magpiid):
    magpifile = fits.open(datapath + "MAGPI"+str(magpiid)+"_minicube.fits")
    psfhdr = magpifile[4].header
    # note that the datacubes have mistake, alpha is beta , beta is alpha
    beta = psfhdr['MAGPI PSF ZBAND MOFFAT ALPHA']
    alpha = psfhdr['MAGPI PSF ZBAND MOFFAT BETA']
    
    weight1, weight2, fwhm1, fwhm2 = mof_to_gauss_threeGau(alpha=alpha, beta=beta, plot=True,magpiid=magpiid)



'''

parent_dir = '/Users/ymai0110/Documents/Blobby3D/v220/minicube/'

fits_doc = fits.open(parent_dir + 'MAGPI1202300176_minicube.fits')
img = fits_doc[4].data[3]
magpiid = 1202300176

'''