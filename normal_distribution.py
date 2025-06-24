#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 11:34:15 2025

@author: ymai0110
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.stats import norm

# Parameters of the normal distribution
mu = 0      # mean
sigma = 1   # standard deviation

# x range
x = np.linspace(-5, 5, 1000)

# Using error function to compute the CDF
cdf_values = 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))

# Using scipy to compute the PDF and CDF for comparison
pdf_values = norm.pdf(x, mu, sigma)
cdf_scipy = norm.cdf(x, mu, sigma)

# Plotting
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, pdf_values, label='PDF', color='blue')
plt.title("Probability Density Function")
plt.xlabel("x")
plt.ylabel("PDF")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, cdf_values, label='CDF (erf)', linestyle='--')
plt.plot(x, cdf_scipy, label='CDF (scipy)', linestyle=':')
plt.title("Cumulative Distribution Function")
plt.xlabel("x")
plt.ylabel("CDF")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Plotting
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, pdf_values, label='PDF', color='blue')
plt.title("Probability Density Function")
plt.xlabel("x")
plt.ylabel("PDF")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x[:-1], np.diff(cdf_values), label='CDF (erf)', linestyle='--')
plt.plot(x[:-1], np.diff(cdf_scipy), label='CDF (scipy)', linestyle=':')
plt.title("Cumulative Distribution Function")
plt.xlabel("x")
plt.ylabel("CDF")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()