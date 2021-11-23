# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 00:16:20 2021

@author: satyam
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import decomposition
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from PIL import Image
from scipy.ndimage import  rotate

Z=np.zeros((10000,4,4))
Z[0:199]=np.array([[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1]])   ##A
Z[200:399]=np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[0,1,1,1]]) ##C
Z[400:599]=np.array([[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,0]]) ##D
Z[600:799]=np.array([[1,1,1,1],[1,0,0,0],[1,1,1,1],[1,0,0,0]]) ##F
Z[800:999]=np.array([[0,1,1,0],[1,0,0,0],[1,0,1,1],[0,1,1,0]]) ##G
Z[1000:1199]=np.array([[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1]]) ##H
Z[1200:1399]=np.array([[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]]) ##I
Z[1400:1599]=np.array([[0,0,0,1],[0,0,0,1],[1,0,0,1],[1,1,1,1]]) ##J
Z[1600:1799]=np.array([[1,0,0,1],[1,0,1,0],[1,1,1,0],[1,0,0,1]]) ##K
Z[1800:1999]=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]]) ##L
Z[2000:2199]=np.array([[1,0,0,1],[1,1,0,1],[1,0,1,1],[1,0,0,1]]) ##N
Z[2200:2399]=np.array([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]]) ##O
Z[2400:2599]=np.array([[1,1,1,1],[1,0,0,1],[1,1,1,1],[1,0,0,0]]) ##P
Z[2600:2799]=np.array([[1,1,1,0],[1,0,1,0],[1,1,1,0],[0,0,1,1]]) ##Q
Z[2800:2999]=np.array([[1,1,1,1],[1,0,0,1],[1,1,1,0],[1,0,0,1]]) ##R
Z[3000:3199]=np.array([[1,1,1,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]) ##T
Z[3200:3399]=np.array([[0,1,1,0],[0,1,0,0],[0,0,1,0],[0,1,1,0]]) ##S
Z[3400:3599]=np.array([[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]]) ##U
Z[3600:3799]=np.array([[0,1,0,1],[0,1,0,1],[0,1,0,1],[0,0,1,0]]) ##V
Z[3800:3999]=np.array([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]]) ##X
Z[4000:4199]=np.array([[0,1,0,1],[0,1,0,1],[0,0,1,0],[0,0,1,0]]) ##Y
Z[4200:4399]=np.array([[1,1,1,0],[0,0,1,0],[0,1,0,0],[0,1,1,1]]) ##Z


Z= shuffle(Z[:4000])
fig, ax = plt.subplots(5,5, figsize = (10,10))
plt.title('Original Images')
plt.tight_layout()
axes = ax.flatten()
for i in range(25):
    axes[i].imshow(Z[i],cmap="Greys")


Siz=Z[0:199] #size track

N_Z=np.zeros((4000,4,4))#For storing noissy image

#Noise_1
Mean=0;
var=0.1
noise1=np.random.normal(Mean,var,Siz.shape)
Z[0:199]=Z[0:199]+noise1
Z[0:199]=Z[0:199]+noise1
Z[0:199]=Z[0:199]+noise1
Z[0:199]=Z[0:199]+noise1
Z[0:199]=Z[0:199]+noise1

#Noise_2
Mean=0;
var=0.2
noise2=np.random.normal(Mean,var,Siz.shape)
Z[200:399]=Z[200:399]+noise2
Z[200:399]=Z[200:399]+noise2
Z[200:399]=Z[200:399]+noise2
Z[200:399]=Z[200:399]+noise2
Z[200:399]=Z[200:399]+noise2

#Noise_3
Mean=0;
var=0.08
noise3=np.random.normal(Mean,var,Siz.shape)
Z[400:599]=Z[400:599]+noise3
Z[400:599]=Z[400:599]+noise3
Z[400:599]=Z[400:599]+noise3
Z[400:599]=Z[400:599]+noise3
Z[400:599]=Z[400:599]+noise3

#Noise_4
Mean=0;
var=0.3
noise4=np.random.normal(Mean,var,Siz.shape)
Z[600:799]=Z[600:799]+noise4
Z[600:799]=Z[600:799]+noise4
Z[600:799]=Z[600:799]+noise4
Z[600:799]=Z[600:799]+noise4
Z[600:799]=Z[600:799]+noise4

#Noise_5
Mean=0;
var=0.06
noise5=np.random.normal(Mean,var,Siz.shape)
Z[800:999]=Z[800:999]+noise5
Z[800:999]=Z[800:999]+noise5
Z[800:999]=Z[800:999]+noise5
Z[800:999]=Z[800:999]+noise5
Z[800:999]=Z[800:999]+noise5
Z[800:999]=Z[800:999]+noise5


N_Z=Z

fig, ax = plt.subplots(5,5, figsize = (10,10))
plt.title('Noisy Image')
plt.tight_layout()
axes = ax.flatten()
for i in range(25):
    axes[i].imshow(N_Z[i],cmap="Greys")

A=N_Z.reshape(4000,16)
####Applying PCA####
pca=PCA(2)
newI=pca.fit_transform(A)
recrI=pca.inverse_transform(newI)


Final=recrI.reshape(4000,4,4)
fig, ax = plt.subplots(5,5, figsize = (10,10))
plt.title('PCA OutPut')
plt.tight_layout()
axes = ax.flatten()
for i in range(25):
    axes[i].imshow(Final[i],cmap="Greys")
    
    
    
nmf = NMF(2)
W = nmf.fit_transform(np.abs(A))
H = nmf.inverse_transform(W)


B=H.reshape(4000,4,4)
fig, ax = plt.subplots(5,5, figsize = (10,10))
plt.title('NMF OutPut')
plt.tight_layout()
axes = ax.flatten()
for i in range(25):
    axes[i].imshow(B[i],cmap="Greys")
    


svd = TruncatedSVD(n_components=5)
svd_trans=svd.fit_transform(A)
svd_op=svd.inverse_transform(svd_trans)
S=svd_op.reshape(4000,4,4)
fig, ax = plt.subplots(5,5, figsize = (10,10))
plt.title('SVD Output')
plt.tight_layout()
axes = ax.flatten()
for i in range(25):
    axes[i].imshow(S[i],cmap="Greys")
    
#Adding rotation

#angle = 90

# for i in range(0,199): # N_Z[i] = rotate (N_Z[i], angle)

# angle = 180

# for i in range (200, 399):

# N_Z[i]= rotate (N_Z[i], angle)

# angle = 270

# for i in range (400,599): # N_Z[i]= rotate (N_Z[i], angle)

# angle = 360

# for i in range (600,799): # N_Z[i] = rotate (N_Z[i], angle)

# angle = 270

# for i in range (800,999): # N_Z[i] = rotate (N_Z[i], angle)


#### for covariance matrix
"""newarray_A_meaned = newarray_A -np.mean(newarray_A)
cov_mat_A = np.cov(newarray_A_meaned)
eigen_values, eigen_vectors = np. linalg.eigh(cov_mat_A)
sorted_index = np.argsort(eigen_values)[::-1] 
sorted_eigenvalue = eigen_values[sorted_index] 
sorted_eigenvectors = eigen_vectors[:,sorted_index]
n_components = 16
eigenvector_subset =sorted_eigenvectors[:,0:n_components]"""

