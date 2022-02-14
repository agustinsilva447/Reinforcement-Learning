import numpy as np
import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax import grad

import matplotlib.pyplot as plt
from functools import reduce
from qgrad.qgrad_qutip import basis, fidelity

def rx(phi):
    return jnp.array([[jnp.cos(phi / 2), -1j * jnp.sin(phi / 2)],
                     [-1j * jnp.sin(phi / 2), jnp.cos(phi / 2)]])

def ry(phi):
    return jnp.array([[jnp.cos(phi / 2), -jnp.sin(phi / 2)],
                     [jnp.sin(phi / 2), jnp.cos(phi / 2)]])

def rz(phi):
    return jnp.array([[jnp.exp(-1j * phi / 2), 0],
                     [0, jnp.exp(1j * phi / 2)]])   

def J():
    I = jnp.array([[1, 0],
                [0, 1]])
    X = jnp.array([[0, 1],
                [1, 0]])    
    I_f = jnp.kron(I, I)
    X_f = jnp.kron(X, X)
    J = jnp.array((1 / jnp.sqrt(2)) * (I_f + 1j * X_f))    
    return J

def J_dag():
    J_dag = jnp.transpose(jnp.conjugate(J()))
    return J_dag

def circuit(params):
    thetax, thetay, thetaz = params
    layer0 = jnp.kron(basis(2, 0), basis(2, 0))
    layer1 = J()
    layer2 = jnp.kron(rx(thetax), rx(thetax))
    layer3 = jnp.kron(ry(thetay), ry(thetay))
    layer4 = jnp.kron(rz(thetaz), rz(thetaz))
    layer5 = J_dag()
    layers = [layer5, layer4, layer3, layer2, layer1]
    unitary = reduce(lambda  x, y : jnp.dot(x, y), layers)
    #unitary = np.dot(layer5,np.dot(layer4, np.dot(layer3, np.dot(layer2, layer1))))    
    return jnp.dot(unitary, layer0)      

params = [1.5704896450042725, 5.727196216583252, 3.9264814853668213]
output = circuit(params)
print("Quantum state output = {}.\n".format(np.round(output,3)))     