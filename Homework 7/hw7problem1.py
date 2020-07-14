# -*- coding: utf-8 -*-
"""
hw7problem1.py
@author: James Cotter
Poisson equation in a rectangle using code from class.
"""

# Poisson Eqn 2D
# using Gauss-Seidel and overrelaxation

import numpy as np

# grid size: make it odd, so it has a definite center
N0 = 201
M0 = 171

# physical origin of the grid
x0 = 0.0
y0 = 0.0

# resolution
eps = 0.01

# physical extent, guarantee identical resolution along x and y
x1 = (N0-1) * eps + x0
y1 = (M0-1) * eps + y0

x = np.linspace(x0, x1, N0)
y = np.linspace(y0, y1, M0)

# mixed boundary coefficients, set now for Dirichlet
cl = 1.0; dl = 0.0
cr = 1.0; dr = 0.0
ct = 1; dt = 0
cb = 1; db = 0

k = 1 # constant coefficient in Poisson's equation

w = 0.0 # overrelaxation parameter

def setDemo1():
    global px, py, coef, demo
    coef = [-1] # -1 / eps_0 * q, scaled
    px = [int(N0 / 2)]
    py = [int(M0 / 2)]
    demo = 1

def setDemo2():
    global px, py, coef, demo
    
    px = []
    py = []
    coef = []
    i = 0
    
    while i < 20:
        px.append(np.random.randint(N0))
        py.append(np.random.randint(M0))
        if np.random.rand(1) < 0.5:
            coef.append(-1)
        else:
            coef.append(1)
        i+=1
        
    demo = 2

    
def rho(x,y):
    """Field density function, expects meshgrid matrices x, y"""
    R = np.zeros(x.shape)
    for i in range(len(px)):
        R[py[i],px[i]] = coef[i] * k
    return R

# boundary functions
def gl(y):
    """boundary function for the left side"""
    return np.zeros(y.shape)

def gr(y):
    """boundary function for the right side"""
    return np.zeros(y.shape)

def gt(x):
    """boundary function for the top side"""
    return np.zeros(x.shape)

def gb(x):
    """boundary function for the bottom side"""
    return np.zeros(x.shape)

def setBoundary(phi):
    """set the values on the boundary according to the current configuration"""
    phi[:, 0] = (dl * phi[:, 1] - eps*ggl) / (dl - cl*eps) # left    
    phi[:,-1] = (dr * phi[:,-2] + eps*ggr) / (dr + cr*eps) # right
    phi[0, :] = (dt * phi[1, :] - eps*ggt) / (dt - ct*eps) # top
    phi[-1,:] = (db * phi[-2,:] + eps*ggb) / (db + cb*eps) # bottom

def getStepDifference(phi1, phi2):
    return np.max(np.abs(phi2[1:-1, 1:-1] - phi1[1:-1, 1:-1]))

def jacobiStep(phi1, phi2):
    
    # apply boundary
    setBoundary(phi1)
    
    # compute next step
    phi2[1:-1, 1:-1] = (1 + w)/4 * (phi1[0:-2, 1:-1] + phi1[2:, 1:-1] + phi1[1:-1, 0:-2] + phi1[1:-1, 2:])
    phi2[1:-1, 1:-1] -= w * phi1[1:-1, 1:-1] + k * eps**2 / 4 * R[1:-1, 1:-1]
    
    return phi1, phi2

def solveJacobi():    
    global R, ggl, ggr, ggt, ggb
    
    # create the mesh, evaluate rho
    X, Y = np.meshgrid(x, y)
    R = rho(X, Y)
    
    # initialize: start with some phi
    phi1 = np.zeros((M0, N0))
    phi2 = np.ones((M0, N0))
    delta = getStepDifference(phi2, phi1)
    
    # evaluate the RHS of the boundary conditions
    ggl = gl(y) # left 
    ggr = gr(y) # right
    ggt = gt(x) # top
    ggb = gb(x) # bottom
    
    it = 0
    # solve
    # terminating conditions: relative error and max iterations
    while (delta/eps > 1e-9 and it < 100000):
        it += 1
        
        # step and swap arrays
        phi2, phi1 = jacobiStep(phi1, phi2)
        
        if it % 10 == 0:
            delta = getStepDifference(phi1, phi2)
    
    print("Finished in %d steps, error=%e" % (it, delta/eps))

    setBoundary(phi1)
    return phi1    
# -----------------------------------------------------------------------------

#setDemo1()
setDemo2()

phi1 = solveJacobi()

# contour levels to draw
levels = np.hstack((np.geomspace(-1e-4, -1e-7, 19), np.geomspace(1e-7, 1e-4, 19)))

# levels on the colorbar
clevels = np.hstack((np.geomspace(-1e-4, -1e-7, 4), np.geomspace(1e-7, 1e-4, 4)))


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

plot = plt.contourf(x, y, phi1, 
             levels=levels,              
             cmap=plt.get_cmap('RdBu'),
             norm = colors.SymLogNorm(linthresh=1e-8, linscale=1e-8,
                                              vmin=-1e-4, vmax=1e-4))

plt.contour(x, y, phi1,
            levels = levels,
            colors='k', linewidths=0.5, linestyles='solid')

plt.xlim(x0, x1)
plt.ylim(y0, y1)

ax = plt.gca()
ax.set_aspect(1)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05) # trick to match colorbar size to Y axis
plt.colorbar(plot, cax = cax, label='$\phi$', extend='both', ticks=clevels)
plt.tight_layout()
plt.savefig('poisson2D_jacobi_%d.png' % demo)
