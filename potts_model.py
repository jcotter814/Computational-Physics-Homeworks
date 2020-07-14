# Potts model, 2D

import numpy as np

# helper class for the ising model implementation
# creates the checkerboard decomposition index patterns
class CheckerBoard2D:
    
    def __init__(self, N, M):
        # create checkerboard indexing for sliced MC updates
        p1 = np.zeros((N,M), dtype=np.bool)
        p1[1::2, ::2] = True
        p1[::2, 1::2] = True
        p2 = ~p1
        rows1, cols1 = np.where(p1)
        rows2, cols2 = np.where(p2)
        
        idx1 = (rows1, cols1)
        idx2 = (rows2, cols2)
        
        # neighbors for each direction
        left1 = (rows1, cols1 - 1)
        right1 = (rows1, (cols1 + 1) % M)
        up1 = (rows1 - 1, cols1)
        down1 = ((rows1 + 1) % N, cols1)
        
        left2 = (rows2, cols2 - 1)
        right2 = (rows2, (cols2 + 1) % M)
        up2 = (rows2 - 1, cols2)
        down2 = ((rows2 + 1) % N, cols2)
        
        self.idx1 = idx1
        self.idx2 = idx2
        
        self.cols1 = cols1
        self.cols2 = cols2
        
        self.rows1 = rows1
        self.rows2 = rows2
        
        self.left1 = left1
        self.right1 = right1
        self.up1 = up1
        self.down1 = down1

        self.left2 = left2
        self.right2 = right2
        self.up2 = up2
        self.down2 = down2
        
class Potts:
    
    def __init__(self, N, M,q):
        """Create a 2D spin lattice with given rows (N) and columns (M)"""
        
        if (N%2 != 0) or (M%2 != 0):
            raise ValueError('Lattice size must be even')
        
        # constants
        self.N = N
        self.M = M
        self.q = q
        self.J = 1.0 # coupling strength
        self.kB = 1.0 # Boltzmann coefficient

        # new random lattice
        self.makeNewLattice()
        
        # private-like members
        self.x = np.arange(M)  # column indices
        self.y = np.arange(N)  # row indices    
        self.i = CheckerBoard2D(N, M)     # indexing scheme
        self.deltaE = np.zeros((N, M))  # temporary array used in mcStep
    
    def makeNewLattice(self):
        """Create a new random spin lattice"""
        self.S = np.random.randint(0, self.q, size=(self.N, self.M))

    def calculateEnergy(self):
        """Calculate the total energy of the current lattice"""
        # neighbors to the left
        e1 = -self.J * np.sum((self.S[:, self.x] == self.S[:, self.x-1]).astype(int))
        
         # neighbors above
        e2 = -self.J * np.sum((self.S[self.y, :] == self.S[self.y-1, :]).astype(int))
        
        return e1 + e2
    
    def energy_diff(self,X,Y,ID):
        i = self.i
        #r = len(Y[i.idx1])
        
        if (ID == 1):
            t1 = ((X[i.left1] == Y[i.idx1]).astype(int) + (X[i.right1] == Y[i.idx1]).astype(int) + (X[i.idx1] == Y[i.up1]).astype(int) + (X[i.idx1] == Y[i.down1]).astype(int))
            #print(t1)
            return t1
        elif (ID == 2):
            t2 = ((X[i.left2] == Y[i.idx2]).astype(int) + (X[i.right2] == Y[i.idx2]).astype(int) + (X[i.idx2] == Y[i.up2]).astype(int) + (X[i.idx2] == Y[i.down2]).astype(int))
            #print(t2)
            return t2
        
    def mcStep(self, T):
        """Compute one MC step (one attempted flip at each site) at the given temperature"""
        
        # generate a set of random numbers
        #Q = np.random.randint(0,high = self.q)*np.ones((self.N, self.M))
        #print("Q is : ",Q)
        Q = np.random.randint(0,self.q,size=(self.N, self.M))
        R = np.random.rand(self.N, self.M)
        #print("R is : ",R)
        
        # local reference to objects
        i = self.i
        S = self.S
        deltaE = self.deltaE
    
        # compute the energy difference of spin flip for sites in idx1
        E1 = -self.J*self.energy_diff(S,S,1)
        E2 = -self.J*self.energy_diff(S,Q,1)
        #print(deltaE)
        deltaE[i.idx1] = E2-E1
        #print(deltaE)
        
        # Metropolis
        accept = (deltaE[i.idx1] < 0.0) | (R[i.idx1] < np.exp(-deltaE[i.idx1] / (self.kB * T)))
    
        # execute accepted spin flips    
        S[i.rows1[accept], i.cols1[accept]] = Q[i.rows1[accept],i.cols1[accept]]
        
        
        # compute the energy difference of spin flip for sites in p2
        E1 = -self.J*self.energy_diff(S,S,2)
        E2 = -self.J*self.energy_diff(S,Q,2)
        #print(deltaE)
        deltaE[i.idx2] = E2-E1
        
        # Metropolis
        accept = (deltaE[i.idx2] < 0.0) | (R[i.idx2] < np.exp(-deltaE[i.idx2] / (self.kB * T)))
    
        # execute accepted spin flips    
        S[i.rows2[accept], i.cols2[accept]] = Q[i.rows2[accept],i.cols2[accept]]
