# -*- coding: utf-8 -*-
"""
hw7problem3.py
@author: James Cotter
2D wave equation in an ellipsoid. I centered it at 0,0 because the question
indicated no offset, so the portion in the lattice is one quarter of the ellipse
Ferenc indicated that it doesn't matter so I left it.
"""
import numpy as np
import matplotlib.pyplot as plt

Lx = 1.0     # system length (m)
c = 1.0     # wave speed
N = 201     # number of points on grid
M = 201
h = 1e-4    # time step

# space grid
x = np.linspace(0, Lx, N)
dx = x[1] - x[0]

# compute height using the same grid spacing as horizontally
Ly = M * dx
y = np.linspace(0, Ly, M)

# origin of the gaussian pulse, in real coordinates
x0 = x[70]
y0 = y[70]

def inbounds(m,n):
    """Defines the points that lie within an ellipse of width 0.7m and height of
    1m"""
    a,b = (0.5*0.7), (0.5*1)
    
    C = np.power(m,2)/a**2 + np.power(n,2)/b**2
    A = C<1
    
    for i in range(len(m)):
        for j in range(len(m)):
            if ~A[i,j]:
                m[i,j] = 0
    
    for i in range(len(m)):
        for j in range(len(m)):
            if ~A[i,j]:
                m[i,j] = 0
    
    return m,n

# compute the mesh grid
x_, y_ = np.meshgrid(x, y)
X,Y = inbounds(x_,y_)

sigma = c * h / dx
sigma2 = sigma**2

# boundary constants for Dirichlet boundary
bl = 0
br = 0
bt = 0
bb = 0

def f(X,Y):
    """initial condition, including boundary points"""
    
    # gaussian pulse
    u = np.exp(-100 * (X - x0)**2) * np.exp(-100 * (Y-y0)**2)
        
    return u

def g(X, Y):
    """initial condition time derivative, interior points only"""
    
    # zero velocity (pluck a string)
    return np.zeros((N-2,M-2))

# boundary condition implementations -----------------------------------------
def boundaryDirichlet(u):
    u[:,0] = bl
    u[:,-1] = br
    u[0,:] = bt
    u[-1,:] = bb
    
    for i in range(len(u)):
        for j in range(len(u)):
            if (X[i,j] == 0):
                u[i,j] = 0

# computational components ----------------------------------------------------
def boundary(u2, u1):
    boundaryDirichlet(u2)
    
def initialize():
    
    # initial condition, including boundary points
    u0 = f(X, Y)    

    # special first step    
    u1 = np.zeros(X.shape)
    u1[1:-1, 1:-1] = u0[1:-1, 1:-1] + 0.5 * sigma2 * (
            u0[0:-2, 1:-1] + u0[2:, 1:-1] +
            u0[1:-1, 0:-2] + u0[1:-1, 2:] -
            4 * u0[1:-1, 1:-1]) + h * g(X, Y)

    # apply b.c.    
    boundary(u1, u0)
    #u1,u0 = inbounds(u1,u0)

    return u0, u1

def step(u0, u1, u2):    
    
    # step formula
    u2[1:-1, 1:-1] = 2 * u1[1:-1,1:-1] + sigma2 * (
            u1[0:-2, 1:-1] + u1[2:, 1:-1] + 
            u1[1:-1, 0:-2] + u1[1:-1, 2:] - 
            4 * u1[1:-1,1:-1]) - u0[1:-1,1:-1]
    
    # b.c. for future steps
    boundary(u2, u1)
    #u2,u1 = inbounds(u1,u0)
    
    
def solveWave(tmax, recordingTimes):
    u0, u1 = initialize()
    u2 = u1.copy()
    
    t = 0
    i = 0
    
    # record some snapshots
    Trecords = []
    
    while(t < tmax):
        # step and rotate time indices
        step(u0, u1, u2)
        u0, u1, u2 = u1, u2, u0
        t += h
        print(t)
        
        # check if we reached the next recording time
        if i < len(recordingTimes) and t >= recordingTimes[i]:
            Trecords.append(u1.copy())
            i += 1  # advance among the recording times
    return Trecords

# --make it go ----------------------------------------------------------------
tmax = 1
times = np.linspace(0, tmax, 500)
records = solveWave(tmax, times)

print('done with computations')

# make a movie ----------------------------------------------------------------
plt.rcParams['animation.ffmpeg_path'] = r'E:\Etc\ffmpeg\bin\ffmpeg.exe'
import matplotlib.animation as animation

# create the initial plot

levels = np.linspace(-0.5, 0.5, 21)
fig, ax = plt.subplots()

c1 = plt.contourf(x, y, records[1], 
             levels=levels,              
             cmap=plt.get_cmap('RdBu'))

c2 = plt.contour(x, y, records[1],
            levels = levels,
            colors='k', linewidths=0.5, linestyles='solid')

ax = plt.gca()
ax.set_aspect(1)
plt.xlabel('x')
plt.ylabel('y')
ax.set_xlim(0,0.4)
ax.set_ylim(0,0.5)
title = plt.title('t=%4.3f' % times[1])
plt.tight_layout()

# define the parts that change over the frames
def anim(i):
    global c1, c2
    
    # remove the old contour artists
    for c in c1.collections:
        c.remove()
    for c in c2.collections:
        c.remove()
    
    c1 = plt.contourf(x, y, records[i], 
                 levels=levels,              
                 cmap=plt.get_cmap('jet'))
    
    c2 = plt.contour(x, y, records[i],
                levels = levels,
                colors='k', linewidths=0.5, linestyles='solid')

    title.set_text('t=%4.3f' % times[i])
    print('frame %d' % i)
    return c1, c2, title

# create the animation
ani = animation.FuncAnimation(fig, anim, interval=50, frames=range(len(times)), blit=False)
#
## save as movie
##ani.save('wave.mp4')
#

#writer = animation.FFMpegWriter(fps=25, bitrate=1800)
#ani.save("wave2D.mp4", writer=writer)


