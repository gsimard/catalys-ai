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

    # |r|^3
    rn3 = rn ** 3

    # Normalize r
    ru = r * 1/rn[:, np.newaxis]

    # Projection of m on unit r
    # m . ru
    d = np.sum(np.multiply(m, ru), 1)

    # B = u0/(4*pi) * (3 ru (m . ru) - m) / |r|^3
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

    # |r|^3
    rn3 = rn ** 3

    # |r|^5
    rn5 = rn ** 5
    
    # B = u0/(4*pi) * (3 r (m . r) - m) / |r|^5
    B = B0 * (3 * np.diag(r) @ np.tile(r, (3,1)) / rn5 - np.identity(3) / rn3)
    
    return tf.convert_to_tensor(B)

# A x = b => return x
def pinv(A, b, reltol=1.0e-6):
    # Compute the SVD of the input matrix A
    s, u, v = tf.svd(A)

    # Invert s, clear entries lower than reltol*s[0].
    atol = tf.reduce_max(tf.abs(s)) * reltol
    s = tf.boolean_mask(s, s > atol)
    s_inv = tf.diag(tf.concat([1. / s, tf.zeros([tf.size(b) - tf.size(s)], dtype=tf.float64)], 0))

    # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
    return tf.matmul(v, tf.matmul(s_inv, tf.matmul(u, tf.reshape(b, [-1, 1]), transpose_a=True)))

# Number of dipoles
Nr = 4

# Positions of moments
r = np.array([
    [0, 2*np.cos(theta), 2*np.sin(theta)] for theta in np.linspace(0,2*pi,Nr,False)])

# Direction of moments: all pointing towards +X
m = 1/Nr * np.cross(r, np.cross([1,0,0], r))

# Direction of moments: circulating around the shaft
#m = 1/Nr * np.cross([1,0,0], r)

# Request the B field at many positions at once
# Number of B measurement points
Np = 128
# Positions of B measurement points
pos = np.array([
    [0.1, 2.1*np.cos(theta + 2*pi/20), 2.1*np.sin(theta + 2*pi/10)] for theta in np.linspace(0,2*pi,Np,False)])


#pos = np.array([5,-3,4])
#B = calculate_B_field(pos - r, m)
#print(B)

# Tensorflow to the rescue !
sess = tf.Session()

# Build matrix of matrices
G = tf.concat([(tf.concat([calculate_B_field_matrix(p - r[i], m[i]) for i in range(0, len(m))], 1)) for p in pos], 0)

#print(sess.run(G))

# Flatten moment list into simple vector
m_flat = tf.reshape(m, [-1])

# b = G*m
b_flat = tf.einsum('nm,m->n', G, m_flat)

# Unflatten B
B2 = sess.run(tf.reshape(b_flat, [-1, 3]))
#print(sess.run(B2))
#print(B2[:,0])

#M2 = sess.run(tf.reshape(pinv(G, b_flat), [-1, 3]))

GP = np.linalg.pinv(sess.run(G))
bb = sess.run(b_flat)

print(m)
print(np.reshape(GP @ bb, (-1,3)))


#import matplotlib.pyplot as plt
#plt.plot(B2[:,0])
#plt.show()
