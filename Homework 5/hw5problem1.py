# -*- coding: utf-8 -*-
"""
hw5problem1.py
Created on Tue Oct  8 22:21:52 2019
@author: James Cotter
Banach matchstick problem.
"""
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
# ----------- Part a ------------- 

def picking(n):
    """Simulates the picking involved in the matchbox problem.
    Inputs:
        n: number of matches
    """
    
    right_pocket = n
    left_pocket = n
    
    while (right_pocket > -1 and left_pocket > -1): #Ensures that he continues
                                                    #picking until he realizes it is empty
        rand = np.random.rand(1)
        if rand > 0.5:
            right_pocket -= 1
        else:
            left_pocket -= 1
        
    if right_pocket == -1:
        return left_pocket
    elif left_pocket == -1:
        return right_pocket
  
n_min = 1
n_max = 1000
Z = 1.96 # 95% confidence interval 
r_avg_array = np.zeros(n_max) #To be filled with the average r value with n = arr[n]
CI_array = np.zeros((2,n_max)) #confidence interval  


for n in range(n_min,n_max):
    
    r_avg = [] #running list of outcomes
    trials = 50 #number of different seeds
    
    for seed in range(trials):    
        np.random.seed(None)
        r_avg.append(picking(n))
    
    average = np.mean(r_avg)
    std_dev = np.std(r_avg)
    CI_array[0][n] = Z*std_dev/np.sqrt(n)
    CI_array[1][n] = -Z*std_dev/np.sqrt(n)
    r_avg_array[n] = average
    

# ------------ Part b -----------
def r_analytic(n):
    """Analytic result for the expectation value of r as a function of n
    Inputs:
        n: int, number of matches initially
    """
    
    bino = int(scipy.special.comb(2*n,n,exact =True))
    num = int(2*n+1)
    denom = int(2**(2*n))
    
    r = (num*((bino*10**16)//denom))/(10**16) -1 #With a hint from Ferenc to avoid too large
                                                 #float values
    
    return r

def stirlings(n):
    """Stirling's approximation to the analytic result
    Inputs:
        n: int, number of initial matches
    """    
    r = ((1+2*n)/(np.sqrt(n)*np.sqrt(np.pi)))-1 #Simplified with some math
    
    return r

#print(stirlings(100))

analytic_array = np.zeros(n_max)
stirlings_array = np.zeros(n_max)
n_array = np.arange(n_max)

for i in range(1,n_max):
    analytic_array[i] = r_analytic(i)
    stirlings_array[i] = stirlings(i)

#analytic_array = r_analytic(n_array)
#stirlings_array = stirlings(n_array)

plt.figure(1)
plt.plot(n_array,r_avg_array,'r-',alpha = 0.5)
plt.plot(n_array,stirlings_array,'b')
plt.fill_between(n_array,r_avg_array+CI_array[0],r_avg_array+CI_array[1],color = 'grey',alpha = 0.5)
plt.plot(n_array,analytic_array,'g')
plt.xlabel('n')
plt.ylabel('Average r')
plt.title('analytic and simulation values for r average')


#The stirling's approximation matches the exact result almost perfectly
#The variation in the average grows with n.
