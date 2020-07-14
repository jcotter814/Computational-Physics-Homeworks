# -*- coding: utf-8 -*-
"""
hw6problem4.py
Created on Sun Nov  3 16:29:03 2019
@author: James Cotter
Predicting an eclipse.
"""

import numpy as np
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
from datetime import datetime,timedelta

def rhs(t,state):
    """Computes the right hand side of the ODE for the 3 body problem.
    """
    
    #Masses
    m_sun = 1.9885e30 # kg
    m_earth = 5.97219e24 # kg
    m_moon = 7.349e22 # kg

    # gravitational constant
    G = 6.67408e-11 # in SI units
    
    #position and velocity vectors
    r_sun = state[:3]
    r_earth = state[3:6]
    r_moon = state[6:9]
    
    v_sun = state[9:12]
    v_earth = state[12:15]
    v_moon = state[15:18]
    
    r_sun_earth = scipy.linalg.norm(r_earth-r_sun)
    r_sun_moon = scipy.linalg.norm(r_moon-r_sun)
    r_earth_moon = scipy.linalg.norm(r_moon-r_earth)
    
    #get derivatives
    a_sun = -G*m_earth*(r_sun-r_earth)/np.power(r_sun_earth,3)-G*m_moon*(r_sun-r_moon)/np.power(r_sun_moon,3)
    a_earth = -G*m_moon*(r_earth-r_moon)/np.power(r_earth_moon,3)-G*m_sun*(r_earth-r_sun)/np.power(r_sun_earth,3)   
    a_moon = -G*m_sun*(r_moon-r_sun)/np.power(r_sun_moon,3)-G*m_earth*(r_moon-r_earth)/np.power(r_earth_moon,3)
    
    dr_sun = v_sun
    dr_earth = v_earth
    dr_moon = v_moon
    
    dr = scipy.concatenate((dr_sun,dr_earth,dr_moon))
    dv = scipy.concatenate((a_sun,a_earth,a_moon))
    
    der = scipy.concatenate((dr,dv))
    
    return der

# conversion to SI
au_to_m = 149597870700  # astronomical units to meters
day_to_sec = 86400
au_per_day_to_m_per_sec = au_to_m / day_to_sec

#initial conditions
sun_x = 0.000000000000000E+00*au_to_m
sun_y = 0.000000000000000E+00*au_to_m
sun_z = 0.000000000000000E+00*au_to_m

earth_x = 8.239124376722546E-01*au_to_m
earth_y = 5.557039479030489E-01*au_to_m
earth_z = -2.990482947808963E-05*au_to_m

moon_x = 8.218574403574972E-01*au_to_m
moon_y = 5.544254599559837E-01*au_to_m
moon_z = 1.689046590750413E-04*au_to_m

sun_vx = 0.000000000000000E+00*au_per_day_to_m_per_sec
sun_vy = 0.000000000000000E+00*au_per_day_to_m_per_sec
sun_vz = 0.000000000000000E+00*au_per_day_to_m_per_sec

earth_vx = -9.904814639176402E-03*au_per_day_to_m_per_sec
earth_vy = 1.420484942640340E-02*au_per_day_to_m_per_sec
earth_vz = -4.016082368418687E-07*au_per_day_to_m_per_sec

moon_vx = -9.589759964131514E-03*au_per_day_to_m_per_sec
moon_vy = 1.366348632018184E-02*au_per_day_to_m_per_sec
moon_vz = -1.903435427371054E-05*au_per_day_to_m_per_sec

state0 = [sun_x,sun_y,sun_z,earth_x,earth_y,earth_z,moon_x,moon_y,moon_z,
          sun_vx,sun_vy,sun_vz,earth_vx,earth_vy,earth_vz,moon_vx,moon_vy,moon_vz]
     
#time parameters       
t0 = 0
tmax = 31536000

time = [t0]
state = np.array(state0)


#Integrate
rk = integrate.RK45(rhs,t0,state0,tmax,rtol = 1e-5)

while(rk.status == 'running'):
    rk.step()
    time.append(rk.t)
    state = np.vstack((state,rk.y))
    #print(state)

#Plot
plt.figure(1)
plt.plot(state[:,3],state[:,4])
plt.title('Trajectory of the earth around the sun')
plt.xlabel('x (meters)')
plt.ylabel('y (meters)')

plt.plot(state[:,0],state[:,1])

#---------Part c------------

ref_date = datetime(2019, 10, 28)

#Integrate
rk = integrate.RK45(rhs,t0,state0,tmax,max_step = 3600,rtol = 1e-5)

while(rk.status == 'running'):
    rk.step()
    time.append(rk.t)
    state = np.vstack((state,rk.y))
    if scipy.linalg.norm(state[6:9]-state[0:3]) < scipy.linalg.norm(state[3:6]-state[0:3]):
       eclipse_time = timedelta(seconds = rk.t)+ref_date
       print(eclipse_time)
       break

#using this I found no eclipses although the question would seem to suggest that
#there should be an eclipse within a year.My method was just to compare the
#distance between the moon and sun to the distance between the earth and sun and
#break/print the date/time if it found an instance where that happened. I'm
#unsure why this didn't work and I ran out of time on the assignment.
