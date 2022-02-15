import numpy as np
from scipy import linalg as LA
import random 
import pickle
pauli = {
        "Sz" : np.array([[1,0],[0,-1]]), 
        "Sx" : np.array([[0,1],[1,0]]),
        "Sy" : np.array([[0,-1j],[1j,0]]),
        "I"  : np.array([[1,0],[0,1]]) 
}
def S(sites,i, op):
    if i == 0:
        return np.kron( pauli[op], np.eye(2**(sites-1)))  
    if i == sites-1:
        return np.kron(  np.eye(2**(sites-1)), pauli[op])
    si = np.kron(np.eye(2**(i)),pauli[op])
    isi = np.kron( si,np.eye(2**(sites-1-i))  )
    return isi 

def IsingHamiltonian(sites,h):
    H = - S(sites, sites-1, "Sx")
    for k in range(sites-1):
        Hij = - S(sites, k, "Sz") @ S(sites, k+1, "Sz") -  h* S(sites,k, "Sx")
        H +=  Hij 
    return H

def thermal(H,beta):
    e_H_beta = LA.expm(-beta*H) 
    Z = np.diagonal(e_H_beta).sum()
    return e_H_beta/Z
L =7
H = IsingHamiltonian(L,1)
rho = thermal( H, 1  )


P = np.diagonal(rho)



def measure_P(Hamiltonian,beta,n):
    rho = thermal(Hamiltonian,beta)
    P =  np.diagonal(rho)
    CP = np.zeros(len(P))
    M = np.zeros(len(P))
    for i  in range(1,len(P)+1):
        CP[i-1] = np.sum(P[:i])
    for k in range(n):
        x = random.uniform(0, 1)
        NCP = np.append(CP,np.array([x]))
        sort = np.argsort(NCP)
        pos = np.where( sort==len(P)  )[0][0]
        M[pos] += 1
    return M

samples = np.array([])

with open('thermal.npy', 'wb') as f:
    for T in np.arange(0.25,1,0.25):
        print( measure_P(H,1/T,10000)  )
        np.save(f, np.array([measure_P(H,1/T,10000) for i in range(1)]))

print("----")
with open('thermal.npy', 'rb') as f:
    for T in np.arange(0.25,1,0.25):
        print( np.load('thermal.npy'))
#print(measure_P(H,1,1000))
