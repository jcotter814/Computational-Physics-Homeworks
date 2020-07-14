# -*- coding: utf-8 -*-
"""
hw6problem2.py
@author: James Cotter
Motion of a charged pendulum over a conducting sheet. Solved for the equations
of motion using the method of images.
"""

import numpy as np
import matplotlib.pyplot as plt

def rhs(theta,state):
    #set parameters
    g = 9.81
    l = 1
    d = 1
    q = 5
    k = 10
    m = 1
    

    theta = state[0]
    omega = state[1]
    dtheta = omega
    
    a = (g/l)*np.sin(theta)
    b = (k*q**2)/(2*(l*(1-np.cos(theta))+d))**2
    c = np.sin(theta)/(m*l)
    
    domega = -a-b*c
    
    return np.array((dtheta,domega))

def Leapfrog_step(rhs, state1, state2, t, h):    
    new_state1 = state1 + h * rhs(t+0.5*h, state2)
    new_state2 = state2 + h * rhs(t+h, new_state1)
    return new_state1, new_state2 

def integrate(rhs, state0, timespan, detail=1):
    """Integrate the given RHS over the given timespan, from state0.
    detail=0 returns only the final state, 
    detail=1 returns the whole trajectory"""
    
    if detail==1:
        records = np.zeros((len(timespan), len(state0)))
        records[0,:] = state0
    
    h = timespan[1] - timespan[0]
    
    # leapfrog: initialize 2 state vectors
    x1 = state0
    
    # using a 2nd-order RK step to initialize
    k1 = 0.5 * h * rhs(timespan[0], state0)
    k2 = 0.5 * h * rhs(timespan[0] + 0.25 * h, state0 + 0.5 * k1)
    x2 = state0 + k2    
    
    # using Euler step to initialize
    #x2 = state0 + 0.5*h*rhs(timespan[0], state0) # half-time steps    
    
    for i, t in enumerate(timespan):
        if i==0: continue
        x1, x2 = Leapfrog_step(rhs, x1, x2, t, h)        
        if detail==1:
            records[i,:] = x1
    
    if detail==1:
        return records
    else:
        return x1
    
#initial state
t0 = 0
state0 = np.array((np.deg2rad(10),1))
state1 = np.array((np.deg2rad(30),1))
state2 = np.array((np.deg2rad(160),2))
state3 = np.array((np.deg2rad(175),1.8))


#-------integrate-----------
time = np.arange(0,10,0.01)
first_state = integrate(rhs,state0,time,1)
second_state = integrate(rhs,state1,time,1)
third_state = integrate(rhs,state2,time,1)
fourth_state = integrate(rhs,state3,time,1)

theta1 = first_state[:,0]
omega1 = first_state[:,1]

theta2 = second_state[:,0]
omega2 = second_state[:,1]

theta3 = third_state[:,0]
omega3 = third_state[:,1]

theta4 = fourth_state[:,0]
omega4 = fourth_state[:,1]

#------------Plotting-------------
plt.figure(1)
plt.plot(time,theta1,'r',label = 'small oscillations')
plt.plot(time,theta2,'b',label = 'big oscillations')
plt.plot(time,theta3,'g',label = 'lower energy unbound')
plt.plot(time,theta4,'k',label ='higher energy unbound')

plt.xlabel('t')
plt.ylabel(r'$\mathrm{\theta}$')
plt.xlim(0,10)
plt.legend()
plt.title(r'$\mathrm{\theta(t)\: for\: different\: trajectories}$')

plt.figure(2)
#set parameters
g = 9.81
l = 1
d = 1
q = 5
k = 5
m = 5

th = np.linspace(-12,12,100)
omeg = np.linspace(-10,10,100)
thet,om = np.meshgrid(th,omeg)

DT = om
DOM = (-g/l)*np.sin(thet)-(k*q**2/(2*(l*(1-np.cos(thet))+d))**2)*np.sin(thet)/(m*l)

#finding separatrix
a = (2/(m*l**2))
b = k*q**2/(2*(l*(1-np.cos(th+np.pi))+d))
c = m*g*(l*(1-np.cos(th+np.pi))+d)
d = (k*q**2/(2*d))+m*g*d

sep = np.sqrt(a*(b+c-d))

ax = plt.gca()
ax.streamplot(th,om,DT,DOM)
ax.plot(thet[0],sep,'r-')
ax.plot(thet[0],-sep,'r-')
ax.set_xlabel(r'$\mathrm{\theta}$')
ax.set_ylabel(r'$\mathrm{\dot{\theta}}$')
ax.set_title('Phase Diagram')




    





