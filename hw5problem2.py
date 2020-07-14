# -*- coding: utf-8 -*-
"""
hw5problem2.py
Created on Thu Oct 10 18:47:43 2019
@author: James Cotter
Integrating over an irregular domain.
"""

import numpy as np
import matplotlib.pyplot as plt

#Definition of ellipse 1
def ellipse1(x,y):
    x0,y0,a,b,phi = 1,1,1,4,np.pi/12 
    
    k = ((x-x0)*np.cos(phi)+(y-y0)*np.sin(phi))**2/(a**2)
    l = ((x-x0)*np.sin(phi)-(y-y0)*np.cos(phi))**2/(b**2)
    
    return k+l

#Definition of ellipse 2
def ellipse2(x,y):
    x0,y0,a,b,phi = -1,-1,2,3,np.pi/4
    
    k = ((x-x0)*np.cos(phi)+(y-y0)*np.sin(phi))**2/(a**2)
    l = ((x-x0)*np.sin(phi)-(y-y0)*np.cos(phi))**2/(b**2)

    return k+l

#Monte Carlo Integration
#I do the integration by placing points uniformly randomly in a known area,
#then checking how many of the total points fall inside the union of the
#two ellipses. The ratio of (known area)*(points in region)/(total points)
#gives the area of the region

#Bounds of outer box 
xmax = 3
xmin = -4
ymax = 5
ymin = -4

#Function to sample uniformly from box 
#From the in class code
def sampleFromBox(n):
    points = np.random.rand(n, 2)
    points[:,0] = points[:,0] * (xmax - xmin) + xmin
    points[:,1] = points[:,1] * (ymax - ymin) + ymin
    return points

#pts = sampleFromBox(10000)
#plt.scatter(pts[:,0],pts[:,1])

#Returns the number of points that are in the ellipses
def is_inside(points):
    e1 = ellipse1(points[:,0],points[:,1])
    e2 = ellipse2(points[:,0],points[:,1])
 
    a = np.nonzero(e1 < 1)
    b = np.nonzero(e2 < 1)
    
    x_points1 = [points[i,0] for i in a]
    x_points2 = [points[i,0] for i in b]
    
    x_intersection = list(set(x_points1[0]) & set(x_points2[0]))
    
    return (len(x_points1[0])+len(x_points2[0]) - len(x_intersection))


#Compute the integral
n = 100000
area_rect = (xmax-xmin)*(ymax-ymin)
points_inside_box = sampleFromBox(n)
num_inside = is_inside(points_inside_box)

integral = area_rect*(num_inside/n)
print('The area is %.4f' %integral)

#Plotting the region
points = 10000
x_rand = 8*np.random.rand(points)-4
y_rand = 9*np.random.rand(points)-4

e1 = ellipse1(x_rand,y_rand)
e2 = ellipse2(x_rand,y_rand)

a = np.nonzero(e1 < 1)
b = np.nonzero(e2 < 1)

x_points1 = [x_rand[i] for i in a]
x_points2 = [x_rand[i] for i in b]

y_points1 = [y_rand[i] for i in a]
y_points2 = [y_rand[i] for i in b]

plt.scatter(x_points1,y_points1,color = 'blue')
plt.scatter(x_points2,y_points2,color = 'blue')




