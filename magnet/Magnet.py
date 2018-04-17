import tensorflow as tf
import numpy as np

import pickle
import time

from math import pi

# import matplotlib.pyplot as plt

# Calculate magnetic fields from a distribution of moments m (each
# evaluated at their own local r, provided by caller) and sum them.
def calculate_B_field(r, m):
    # Get norm of r
    rn = np.linalg.norm(r, 2, 1)

    # |r|³
    rn3 = rn ** 3

    # Normalize r
    ru = r * 1/rn[:, np.newaxis]

    # Projection of m on unit r
    # m . ru
    d = np.sum(np.multiply(m, ru), 1)

    # B = u0/(4*pi) * (3 ru (m . ru) - m) / |r|³
    u0 = 4*pi*1e-7
    B = np.sum(u0/(4*pi) * (3*ru*d[:,np.newaxis] - m) / rn3[:,np.newaxis], 0)
#    B = u0/(4*pi) * (3*ru*d[:,np.newaxis] - m) / rn3[:,np.newaxis]

    return B

# Calculate magnetic field at origin (0,0,0) given a distribution of
# moments m localized at r
def calculate_B_field_matrix(r, m):
    u0 = 4*pi*1e-7
    B0 = u0 / (4*pi)

    # Get norm of r
    rn = np.linalg.norm(r)

    # |r|³
    rn3 = rn ** 3

    # |r|^5
    rn5 = rn ** 5
    
    # B = u0/(4*pi) * (3 r (m . r) - m) / |r|^5
    B = B0 * (3 * np.diag(r) @ np.tile(r, (3,1)) / rn5 - np.identity(3) / rn3)
    
    return tf.convert_to_tensor(B)

# Number of dipoles
N = 4

# Positions of moments
r = np.array([
    [0, 2*np.cos(theta), 2*np.sin(theta)] for theta in np.linspace(0,2*pi,N,False)])

# Direction of moments: all pointing towards +X
m = 1/N * np.cross(r, np.cross([1,0,0], r))

# Direction of moments: circulating around the shaft
#m = 1/N * np.cross([1,0,0], r)

pos = np.array([5,-3,4])
B = calculate_B_field(pos - r, m)
print(B)

# Tensorflow to the rescue !
sess = tf.Session()

# Request the B field at many positions at once
pos = np.array([[0,0,0], pos])

# Build matrix of matrices
G = tf.concat([(tf.concat([calculate_B_field_matrix(p - r[i], m[i]) for i in range(0, len(m))], 1)) for p in pos], 0)

#print(sess.run(G))

# Flatten moment list into simple vector
m_flat = tf.reshape(m, [-1])

b_flat = tf.einsum('nm,m->n', G, m_flat)

#B2 = tf.reduce_sum(tf.reshape(b_flat, [-1, 3]), 0)

B2 = tf.reshape(b_flat, [-1, 3])

print(sess.run(B2))
