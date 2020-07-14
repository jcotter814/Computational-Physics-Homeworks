# -*- coding: utf-8 -*-
"""
hw5problem3.py
Created on Fri Oct 11 21:00:41 2019
@author: James Cotter
Importance sampling method of computing an integral.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

def func(x):
    """The function to be integrated"""
    return (np.sin(1/x))**2

#Making the plot
x = np.linspace(-10,10,1000)
y = func(x)
plt.figure(1)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('f(x)')

#Computing the integral using method from class

#Normalization of gaussian is error function
def weight_integral(a,b,sigma):
    r = 0.5*(scipy.special.erf(b*sigma/np.sqrt(2)) - scipy.special.erf(a*sigma/(np.sqrt(2))))/sigma**2 
    return r

#print(weight_integral(-10,10,25))

#Gaussian
def weight_function(x,sigma):
    return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-x**2/(2*sigma**2))

#print(weight_function(1,5))

#Sampling from gaussian
def weight_samples(count,a,b):
    result = np.random.normal(size = count)
    indices = np.nonzero((result <a) | (result >b))
    count = len(indices[0])
    
    while count > 0:
        result[indices] = np.random.normal(size = count)
        indices = np.nonzero((result <a) | (result >b))
        count = len(indices[0])
    return result

#doing the integral
def importance_sampling(func,weight_function,weight_integral,weight_samples,a,b,samples,sigma):
    
    #use chunking method from class
    chunkSize = 1000000
    fullChunkCount = samples//chunkSize
    lastChunkSize = samples - fullChunkCount*chunkSize
    
    norm = weight_integral(a,b,sigma) #normalize
    integral = 0.0
    
    for i in range(fullChunkCount):
        x = weight_samples(chunkSize,a,b)
        fx = func(x)
        wx = weight_function(x,sigma)
        integral += np.mean(fx/wx)*norm
    
    if lastChunkSize > 0:
        x = weight_samples(lastChunkSize,a,b)
        fx = func(x)
        wx = weight_function(x,sigma)
        integral += np.mean(fx/wx)*norm*lastChunkSize/chunkSize
    
    integral/= fullChunkCount + lastChunkSize/chunkSize
    return integral

sigma = 1
sigma1_values = []
for s in range(9):
    r =importance_sampling(func,weight_function,weight_integral,weight_samples,-10,10,10**s,sigma)
    sigma1_values.append(r)
    print('samples: 10^%d\tintegral: %e' % (s,r))

sigma = 5
sigma5_values = []
for s in range(9):
    t =importance_sampling(func,weight_function,weight_integral,weight_samples,-10,10,10**s,sigma)*10
    sigma5_values.append(t)
    print('samples: 10^%d\tintegral: %e' % (s,t))
    
sigma = 25
sigma25_values = []
for s in range(9):
    u =importance_sampling(func,weight_function,weight_integral,weight_samples,-10,10,10**s,sigma)*50
    sigma25_values.append(u)
    print('samples: 10^%d\tintegral: %e' % (s,u))

#Plotting
x = [10**i for i in range(9)]
exact = 2.94181*np.ones(9)


plt.figure(2)
line1, = plt.semilogx(x,sigma1_values,color = 'blue',marker = 'o',label='sigma =1')
line2, = plt.semilogx(x,exact,'--',label = 'exact')
line3, = plt.semilogx(x,sigma5_values,color = 'red',marker = 'D',label = 'sigma = 5')
line4, = plt.semilogx(x,sigma25_values,color = 'green',marker = 'x',label = 'sigma =25')
plt.legend(handles = [line1,line2,line3,line4])

plt.xlabel('x')
plt.ylabel('integral')
plt.title('Integral of sin^2(1/x) using different samplings')

print('Converges fastest when sigma = 5')

    
    
    
    
    