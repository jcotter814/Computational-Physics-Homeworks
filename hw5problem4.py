# -*- coding: utf-8 -*-
"""
hw5problem4.py
Created on Sat Oct 12 15:42:02 2019
@author: James Cotter
Markov chain monte carlo.
"""
# MC simulation, Ising model

import numpy as np
import matplotlib.pyplot as plt


# constants
J = 1.0 # coupling strength
kB = 1.0 # Boltzmann coefficient

nT = 41 # number of steps in temperature
Temps = np.linspace(0.5, 5, nT) # temperatures to evaluate

eqSteps = 1000 # number of MC steps for reaching equilibrium
samplingSteps = 1000 # number of MC samples to run for sampling quantities

# create a new random spin lattice
def initialize():    
    return (np.round(np.random.rand(height, width)) - 0.5) * 2.0

# calculate the energy of the given lattice
def calculateEnergy(s):
    e1 = -J * np.sum(s[:, x] * s[:, (x+1) % width]) # neighbors to the right
    e2 = -J * np.sum(s[y, :] * s[(y+1) % height, :]) # neighbors below
    return e1 + e2

# calculate the magnetization of the given lattice
def calculateMagnetization(s):
    return np.sum(s)

# compute the change of energy if we were to flip the spin at [iy, ix]
def energyDifference(s, ix, iy):    
    nnx = nx[iy, ix, :] # x indices of neighbors
    nny = ny[iy, ix, :] # y indices of neighbors
    
    s0 = s[iy, ix] # the original spin
    return np.sum(s[nny, nnx] * s0) * J * (2.0) # change in energy

# one MC step is one flip attempt on all cells
def mcStep(s, T):    
    for iy in range(height):
        for ix in range(width):
            dE = energyDifference(s, ix, iy) # how would the energy change
            if (dE < 0.0) | (np.random.rand() < np.exp(-dE / (kB * T))):
                s[iy, ix] *= -1.0  # accept flip

# mc simulation for the desired quantities
def monteCarlo():
    # allocate space for the results (one per temperature)
    E = np.zeros(nT) # energy
    M = np.zeros(nT) # abs magnetization
    C = np.zeros(nT) # heat capacity
    X = np.zeros(nT) # susceptibility
    
    # simulate for each temperature
    for i in range(nT):
        T = Temps[i]
        print('T = %f' % T)
        
        s = initialize()
        meanE = 0 # local varaibles for accumulating samples
        meanM = 0
        meanAbsM = 0
        meanE2 = 0
        meanM2 = 0
        
        # reach equilibrium first
        for j in range(eqSteps):
            mcStep(s, T)
            
        # take samples now
        for j in range(samplingSteps):
            mcStep(s, T)
            
            # take samples of interested quantities
            energy = calculateEnergy(s)
            magnetization = calculateMagnetization(s)
            
            # accumulate for averaging
            meanE += energy
            meanM += magnetization
            meanAbsM += np.abs(magnetization)
            meanE2 += energy ** 2
            meanM2 += magnetization ** 2
        
        # compute averages over time    
        meanE /= samplingSteps
        meanM /= samplingSteps
        meanAbsM /= samplingSteps
        meanE2 /= samplingSteps
        meanM2 /= samplingSteps
        
        # store the measured quantities for the current temperature
        E[i] = meanE / (width * height)
        M[i] = meanAbsM / (width * height)
        X[i] = (meanM2 - meanM ** 2) / (kB * T * width * height)
        C[i] = (meanE2 - meanE ** 2) / (kB * T * T * width * height)
    return E, M, X, C


E_array = np.zeros((4,nT))
M_array = np.zeros((4,nT))
X_array = np.zeros((4,nT))
C_array = np.zeros((4,nT))

i = 5
k = 0

while i <= 40:
    
# lattice size
    height = i
    width = i

# precompute indices
    x = np.arange(width)
    y = np.arange(height)

# precomputed neighbor indices
    nx = np.zeros((height, width, 4), dtype=np.int) # x index of neighbors
    ny = np.zeros((height, width, 4), dtype=np.int) # y index of neighbors
    for j in range(height):
        for i in range(width):
            nx[j, i, :] = [i, i, (i-1)%width, (i+1)%width]
            ny[j, i, :] = [(j-1)%height, (j+1)%height, j, j]

# run the simulation
    E, M, X, C = monteCarlo()
    
    E_array[k] = E
    M_array[k] = M
    X_array[k] = X
    C_array[k] = C
    
    i *= 2
    k +=1

# plot the results
fig = plt.figure(figsize = (12,8))

ax1 = fig.add_subplot(2, 2, 1)
plt.scatter(Temps, E_array[0],color = 'red',label = '5x5')
plt.scatter(Temps,E_array[1], color = 'blue',label = '10x10')
plt.scatter(Temps,E_array[2],color = 'green',label = '20x20')
plt.scatter(Temps,E_array[3],color = 'black',label = '40x40')


plt.xlabel('Temperature (T)', fontsize=10)
plt.ylabel('Energy (E/N)', fontsize=10)
plt.title('Energy vs Temperature')

ax2 = fig.add_subplot(2, 2, 2)
plt.scatter(Temps, M_array[0],color = 'red',label = '5x5')
plt.scatter(Temps,M_array[1], color = 'blue',label = '10x10')
plt.scatter(Temps,M_array[2],color = 'green',label = '20x20')
plt.scatter(Temps,M_array[3],color = 'black',label = '40x40')

box = ax2.get_position()
ax2.set_position([box.x0,box.y0,box.width*1,box.height])

ax2.legend(bbox_to_anchor=(1.02,0.5),loc = 'center left',
           ncol = 1,title = 'Lattice Size',borderaxespad = 0.)
plt.xlabel('Temperature (T)', fontsize=10)
plt.ylabel('Magnetization (M/N)', fontsize=10)
plt.title('Magnetization vs Temperature')

ax3 = fig.add_subplot(2, 2, 3)
plt.scatter(Temps, C_array[0],color = 'red',label = '5x5')
plt.scatter(Temps,C_array[1], color = 'blue',label ='10x10')
plt.scatter(Temps,C_array[2],color = 'green',label = '20x20')
plt.scatter(Temps,C_array[3],color = 'black',label ='40x40')


plt.xlabel('Temperature (T)', fontsize=10)
plt.ylabel('Heat capacity (C/N)', fontsize=10)
plt.title('Heat Capacity vs Temperature')

ax4 = fig.add_subplot(2, 2, 4)
plt.scatter(Temps, X_array[0],color = 'red',label = '5x5')
plt.scatter(Temps,X_array[1], color = 'blue',label ='10x10')
plt.scatter(Temps,X_array[2],color = 'green',label = '20x20')
plt.scatter(Temps,X_array[3],color = 'black',label = '40x40')


plt.xlabel('Temperature (T)', fontsize=10)
plt.ylabel('Magnetic susceptibility (X/N)', fontsize=10)
plt.title('Magnetic Susceptibility vs Temperature')

fig.subplots_adjust(hspace =0.5)

fig.savefig('ising.png')
