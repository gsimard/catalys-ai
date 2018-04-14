import tensorflow as tf
import numpy as np

import pickle
import time

from math import pi

# import matplotlib.pyplot as plt

# Calculate magnetic field at origin (0,0,0) given a distribution of
# moments m localized at r
def calculate_B_field(r, m):
    # Get norm of r
    rn = np.linalg.norm(r, 2, 1)

    # |r|³
    rn3 = rn ** 3

    # Normalize r
    ru = r * 1/rn[:, np.newaxis]

    # Projection of m on unit r
    # m . ru
    d = np.sum(np.multiply(m, ru),1)

    # B = u0/(4*pi) * (3 ru (m . ru) - m) / |r|³
    u0 = 4*pi*1e-7
    B = np.sum(u0/(4*pi) * (3*ru*d[:,np.newaxis] - m) / rn3[:,np.newaxis], 0)

    return B

# Number of dipoles
N = 10000

# Positions of moments
r = np.array([
    [0, np.cos(theta), 1.1*np.sin(theta)] for theta in np.linspace(0,2*pi,N,False)])

# Direction of moments
m = 1/N * np.cross([1,0,0], r)


pos = np.array([1,1,0.9])
B = calculate_B_field(r - pos, m)
print(B)
