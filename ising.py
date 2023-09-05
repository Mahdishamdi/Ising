import numpy as np
import matplotlib.pyplot as plt

s = 1 #Spins Coefficient
k = 1 #Boltzmann Coefficient



def make_lattice(n):
    '''Estabilish an n by n lattice by random spin in each position.'''

    spin = [-1,1]
    lattice = np.random.choice(spin,(n,n))
    return lattice


#@jit(nopython=True)
def energy(lattice):
    '''Calculate the total energy of our lattice for our specific hamiltonian (without a magnetic field).'''

    energy = 0
    n = len(lattice)
    for i in range(n):
        for j in range(n):
            energy += -s*0.5*lattice[i,j]*((lattice[(i+1)%n,j])+(lattice[(i-1),j])+(lattice[i,(j+1)%n])+(lattice[i,(j-1)]))
    return energy


#@jit(nopython=True)
def spin_inversion(lattice, tempreture):
    '''change spin of a random place in our lattice if the total energy of our lattice decreases in given tempreture.'''
    
    n = len(lattice)
    i,j = np.random.randint(n),np.random.randint(n)
    delta_energy = -2*(-s*lattice[i,j]*((lattice[(i+1)%n,j])+(lattice[(i-1),j])+(lattice[i,(j+1)%n])+(lattice[i,(j-1)])))
    if (delta_energy < 0) or (np.random.rand() < np.exp(-delta_energy/(k*tempreture))):
        lattice[i,j] = - lattice[i,j]
    return lattice

def iqnore_local_minimums(lattice, tempreture):
    n = len(lattice)
    for i in range(n):
        for j in range(n):
            new_lattice=spin_inversion(lattice, tempreture)
    return new_lattice
        
    
    
    
def magnetization(lattice):
    '''calculate the average magnetization of our lattice'''
    
    mag = np.sum(lattice)
    
    return mag

size = 8
step = 60000
temp = 3.526


e=[]
m=[]
Data=[]
S=np.zeros((step,size,size))
new_lattice = make_lattice(size)


for i in range(step):
    S[i]= iqnore_local_minimums(new_lattice, temp)
    #print('S',average_magnetization(S[i]))
    #print(S[i])
    e.append(energy(S[i]))
    m.append(magnetization(S[i]))
    Data.append(S[i])
    
    #print('data',average_magnetization(Data[i]))


arr=np.array(Data)
arr[arr == -1] = 0
reshapedData=arr.reshape(step,size**2)


np.save("Spin-8*8-data-60k",reshapedData)
np.save("Untrained_En_60k",e)
np.save("Untrained_Mag_60k",m)
