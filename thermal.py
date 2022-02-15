from scipy import linalg as LA
from FSS_dyn.ED_b import *
import deepdish as dd
import numba
from s_const import *
from multiprocessing import Pool
#	**********  BASIC MATRIX DEFINITIONS  **********
#
sigma_x = np.array( [ [ 0. , 1. ] , [ 1. , 0. ] ] )
N = np.array( [ [ 0. , 0. ] , [ 0. , 1. ] ] )
sigma_z = np.array( [ [ 1. , 0. ] , [ 0. , -1. ] ] )
Id = np.array( [ [ 1. , 0. ] , [ 0. , 1. ] ] )
sigma_y = np.array([[0, -1j], [1j, 0]])
Pj = np.array( [ [ 1. , 0. ] , [ 0. , 0. ] ] )
nf = np.array([[1, 0],[0, 0]]) ###  local fermions number nf = 1-sz ###

pauli_dict = {
    'X': sigma_x,
    'Y': sigma_y,
    'Z': sigma_z,
    'I': Id
}
names=['I','X','Y','Z']

observables_dict={}

def pauli_op(sites):
 paulis2=[]
 names_pauli2=[]
 for i in names:
    for j in names:
     names_pauli2.append(i+str(sites[0])+j+str(sites[1]))
     paulis2.append(two_ff(pauli_dict[i],pauli_dict[j],sites[0],sites[1],l))
 return paulis2, names_pauli2
gamma = get_basis(128)
print(gamma)
#Ising Hamiltonian
H = -0.5*H1f(sigma_x, L) - 0.5*h* H3srdef(sigma_z, sigma_z, L)
for beta in np.arange(0,1,0.01)
    rho = LA.expm(-beta*H)/np.diagonal(H).sum
    for k in range(12)
    np.diagonal(gamma@rho).sum
    








