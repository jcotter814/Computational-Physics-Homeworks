# MC simulation, potts model
# Using the Checkerboard decomposition implementation

import potts_model as P
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

nT = 41 # number of steps in temperature
Temps = np.linspace(0.5, 1.5, nT) # temperatures to evaluate

#nT = 1
#Temps = [2]

eqSteps = 5000
samplingSteps = 5000

def monteCarlo(q_i):
    
    model = P.Potts(10,10,q_i)
    
    # allocate space for the results (one per temperature)
    E = np.zeros(nT) # energy
    C = np.zeros(nT) # heat capacity
    
    # simulate for each temperature
    for i in range(nT):
        T = Temps[i]
        print('T = %f' % T)
        
        #model.makeNewLattice()
        meanE = 0 # local varaibles for accumulating samples
        meanE2 = 0
        
        # reach equilibrium first
        for j in range(eqSteps):
            model.mcStep(T)
            
        # take samples now
        for j in range(samplingSteps):
            model.mcStep(T)
            
            # take samples of interested quantities
            energy = model.calculateEnergy()
            
            # accumulate for averaging
            meanE += energy
            meanE2 += energy ** 2
        
        # compute averages over time    
        meanE /= samplingSteps
        meanE2 /= samplingSteps
        
        # store the measured quantities for the current temperature
        size = np.prod(model.S.shape)
        E[i] = meanE / size
        C[i] = (meanE2 - meanE ** 2) / (model.kB * T**2 * size)
    return E, C

results = []
qmin = 2
qmax = 5

for i in range(qmin,qmax):
    
    start = timer()
    # run the simulation
    E, C = monteCarlo(i)
    results.append([E,C])
    end = timer()
    print("Elapsed: %f seconds" % (end - start))
    
fig = plt.figure(figsize = (8,4))
ax1 = fig.add_subplot(1,2,1)

for i in range(0,len(results)):
    #plot the results
    plt.plot(Temps, results[i][0], '-o',label = "q = %.1f" %(i+qmin))
    plt.xlabel('Temperature (T)', fontsize=10)
    plt.ylabel('Energy (E/N)', fontsize=10)
plt.legend()

ax3 = fig.add_subplot(1,2,2)

for i in range(0,len(results)):
    plt.plot(Temps, results[i][1], 'o-',label = "q = %.1f" %(i+qmin))
    plt.xlabel('Temperature (T)', fontsize=10)
    plt.ylabel('Heat capacity (C/N)', fontsize=10)
    
fig.tight_layout()
#fig.savefig('stats.png')
plt.legend()
