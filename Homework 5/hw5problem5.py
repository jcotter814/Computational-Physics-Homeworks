# -*- coding: utf-8 -*-
"""
hw5problem5.py
Created on Sat Oct 12 18:45:44 2019
@author: James Cotter
Simulated annealing to find the minima of a 3D function by adopting the in class
method for 2D functions.
"""
import numpy as np
import matplotlib.pyplot as plt

def objective(x,y):
    """Function to be minimized
    Inputs:
        x,y: floats
    """
    return np.sin(x*y)+0.05*(x-2)**2+0.05*(y-1)**2

#Temperature Parameters
T0 = 50
Tf = 5e-2
Tstep = 0.95

#Fine graining
dx = 1e-1
dy = 1e-1

#Function bounds
xmin = -5
xmax = 5
ymin = -5
ymax = 5


def epochLength(T):
    return ((np.log(T) - np.log(T0)) / (np.log(Tf) - np.log(T0)) * 30 + 10).astype(np.int)

def anneal(): 
    x = np.random.rand() * (xmax - xmin) + xmin #Random x value
    y = np.random.rand() * (ymax - ymin) + ymin #Random y value
    
    #Evaluate the function
    fxy = objective(x,y)
    
    #Set it as global minima
    globalMinF = fxy 
    globalMinX = x
    globalMinY = y
    
    # compute the temperatures and the epochs for each temperature
    tSteps = np.round(np.log(Tf / T0) / np.log(Tstep))
    temps = np.geomspace(T0, Tf, tSteps)
    lengths = epochLength(temps)
    
    # allocate arrays for recording the trajectory
    totalSteps = np.sum(lengths)
    xx = np.zeros(totalSteps)
    yy = np.zeros(totalSteps)
    zz = np.zeros(totalSteps)
    
    step = 0 # counter
    
    # iterate over temperatures    
    for j, T in enumerate(temps):
        
        # start each temperature from the global minimum so far
        x = globalMinX
        y = globalMinY
        fxy = globalMinF
        
        # steps of an epoch
        for i in range(lengths[j]):
            # propose a random step
            newX = x + np.random.randn() * dx * T
            newY = y + np.random.randn()*dy*T
            
            # make sure we stay in the function domain
            newX = np.clip(newX, xmin,xmax)
            newY = np.clip(newY,ymin,ymax)
            
            # evaluate the function
            newF = objective(newX,newY)
            
            # Metropolis: accept or reject
            if (newF < fxy) | (np.random.rand() < np.exp((fxy - newF) / T)):
                # accept
                x = newX
                y = newY
                fxy = newF
                
                #check if we have new global minimum
                if fxy < globalMinF:
                    globalMinF = fxy
                    globalMinX = x
                    globalMinY = y
            
            # record trajectory
            xx[step] = x
            yy[step] = y
            zz[step] = fxy
            
            # count the steps, show progress
            step +=1
            if (step % 100 == 0):
                print('T = %f' % T)
                
    return globalMinX, globalMinY,globalMinF, xx, yy, zz 
    
# run it
minX, minY, minF, xx, yy, zz = anneal()
print('minimum: %f,%f\tfunction value: %f' % (minX,minY,minF))

#Plotting
x = np.linspace(-5,5,1000)
y = np.linspace(-5,5,1000)
X,Y = np.meshgrid(x,y)

obj_func = objective(X,Y)

Z = np.flip(obj_func,0)
plt.imshow(obj_func,extent = [-5,5,-5,5],origin = 'lower',cmap = 'jet')
plt.contour(X,Y,Z,np.linspace(-1,4,20),colors = 'black',linewidths = 0.5,linestyles = 'solid')
plt.plot(minX,minY,marker = 'X')
plt.show()

    
