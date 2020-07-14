# -*- coding: utf-8 -*-
"""
hw6problem1.py
@author: James Cotter
Solving Rossler ODEs.
"""

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integr

# initial condition
state0 = [0,-5,0]
h = 0.01

def rhs(t,state):
    """The right hand side of the ODE. Doesn't depend explicitly on t but
    it is customarily included."""
    
    #Set Parameters
    a = 0.38
    b = 0.3
    c = 4.82

    x = state[0]
    y = state[1]
    z = state[2]
    
    dx = -y-z
    dy = x+a*y
    dz = b-c*z+x*z
    
    return np.array([dx,dy,dz])

def RK4_step(rhs, state, t, h):
    k1 = h * rhs(t, state)
    k2 = h * rhs(t + 0.5*h, state + 0.5*k1)
    k3 = h * rhs(t + 0.5*h, state + 0.5*k2)
    k4 = h * rhs(t + h, state + k3)
    return state + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0

def integrate(rhs, state0, timespan, detail=1):
    """Integrate the given RHS over the given timespan, from state0.
    detail=0 returns only the final state, 
    detail=1 returns the whole trajectory"""
    
    if detail==1:
        records = np.zeros((len(timespan), len(state0)))
        records[0,:] = state0
    
    x = state0
    h = timespan[1] - timespan[0]
    for i, t in enumerate(timespan):
        if i==0: continue    
        x = RK4_step(rhs, x, t, h)
        if detail==1:
            records[i,:] = x
    
    if detail==1:
        return records
    else:
        return x

    
# integration
time_long = np.arange(0, 200, h) # long timespan
state_long = integrate(rhs, state0, time_long, 1)

#Plotting
fig1 = plt.figure(1)
ax1 = fig1.gca(projection = '3d')
ax1.plot(state_long[:,0],state_long[:,1],state_long[:,2])
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Rossler ODEs')

fig,(ax2,ax3,ax4) = plt.subplots(3,sharex = True)

ax2.plot(time_long,state_long[:,0])
ax2.set_xlim(xmin = 100,xmax = 200)
ax2.set_xlabel('t')
ax2.set_ylabel('x(t)')
ax2.set_title('Trajectory Along Each Axis in [100,200]')

ax3.plot(time_long,state_long[:,1])
ax3.set_xlim(xmin = 100,xmax = 200)
ax3.set_xlabel('t')
ax3.set_ylabel('y(t)')

ax4.plot(time_long,state_long[:,2])
ax4.set_xlim(xmin = 100,xmax = 200)
ax4.set_xlabel('t')
ax4.set_ylabel('z(t)')

#It looks like x and y are mirror images of each other and z is the sum of 
#their magnitudes.


