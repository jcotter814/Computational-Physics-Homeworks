# -*- coding: utf-8 -*-
"""
hw6problem3.py
@author: James Cotter
Poincare section of a dynamical system.
Based on the system in the paper:
C. Letellier, R. Gilmore, Poincare sections for a new three-dimensional toroidal attractor
Produces some very nice plots.
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import bisect

def rhs(t,state):
    """The right hand side of the differential equation.
    Inputs:
        t: array, time
        state: array, state vector
    """
    
    #set parameters
    a,c,d,e,f,k = (41,11/6,0.16,0.65,20,55)
    
    #State
    x = state[0]
    y = state[1]
    z = state[2]
    
    #Define 
    dx = a*(y-x)+d*x*z
    dy = k*x+f*y-x*z
    dz = c*z+x*y-e*x**2
    
    return np.array([dx,dy,dz])

#------------Part a--------------
#initial conditions
state0 = [1,2,3]
t0 = 0
tmax = 100

time = [t0]
state = np.array(state0)

rk = integrate.RK45(rhs,t0,state0,tmax,rtol = 1e-9)

while(rk.status == 'running'):
    rk.step()
    time.append(rk.t)
    state = np.vstack((state,rk.y))

plt.figure(1)
plt.scatter(state[:,0],state[:,1],s = 0.1,cmap = 'jet',c = state[:,2])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectory')

#----------Part b-----------------

def integrate_(state0,t0,tmax):
    """Integrates ODE from a given initial state and returns the final state.
    """
    
    rk = integrate.RK45(rhs,t0,state0,tmax,rtol = 1e-9)

    while(rk.status == 'running'):
        rk.step()
        
    return rk.y

def poincare_crossing(state0,state1):
    """Determines if their is a poincare crossing of the y = 0 plane between
    two points of the trajectory x0 and x1.
    """
    
    if np.sign(state0[1]) != np.sign(state1[1]):
        return 1
    else:
        return 0

def get_crossing(state0,t0,tmax):
    """Finds the location of the trajectory's intersection with the plane y = 0.
    """
    def f(t):
        return integrate_(state0,t0,t)[1]
    crossing_t = bisect(f,t0,tmax)
    return integrate_(state0,t0,crossing_t)

#initial conditions
state0 = [1,2,3]
t0 = 0
tmax = 100

time = [t0]
cross = np.array([0,0,0])
state = np.array(state0)

rk = integrate.RK45(rhs,t0,state0,tmax,rtol = 1e-9)

while(rk.status == 'running'):
    rk.step()   
    if poincare_crossing(rk.y_old,rk.y):
        cross = np.vstack((cross,get_crossing(rk.y_old,rk.t_old,rk.t)))
    time.append(rk.t)
    state = np.vstack((state,rk.y))

plt.figure(2)
plt.scatter(cross[:,0],cross[:,2])
plt.title('Poincare Crossings')
plt.xlabel('x')
plt.ylabel('z')
    

    
