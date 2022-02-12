import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax import grad

import matplotlib.pyplot as plt
from functools import reduce
from qgrad.qgrad_qutip import basis, expect

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
    I_f = jnp.array([[1, 0],
                [0, 1]]) 
    I = jnp.array([[1, 0],
                [0, 1]])
    X_f = jnp.array([[0, 1],
                [1, 0]]) 
    X = jnp.array([[0, 1],
                [1, 0]])    
    I_f = jnp.kron(I_f, I)
    X_f = jnp.kron(X_f, X)

    J = jnp.array(1 / jnp.sqrt(2) * (I_f + 1j * X_f))    
    return J

def J_dag():
    J_dag = jnp.conjugate(J()).transpose()
    return J_dag
                         
def circuit(params):
    thetax, thetay, thetaz = params
    layer0 = jnp.kron(basis(2, 0), basis(2, 0))
    layer1 = J()
    layer2 = jnp.kron(rx(thetax), rx(thetax))
    layer3 = jnp.kron(ry(thetay), ry(thetay))
    layer4 = jnp.kron(rz(thetaz), rz(thetaz))
    layer5 = J_dag()
    layers = [layer1, layer2, layer3, layer4, layer5]
    unitary = reduce(lambda  x, y : jnp.dot(x, y), layers)
    return jnp.dot(unitary, layer0)  

def cost(params):
    op = jnp.array([[0, 0,   0,   0],
                    [0, 0.5, 0.5, 0],
                    [0, 0.5, 0.5, 0],
                    [0, 0,   0,   0]])
    state = circuit(params)
    return -jnp.real(expect(op, state))

# fixed random parameter initialization
init_params = [jnp.pi/8, jnp.pi/8, jnp.pi/8]
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)
opt_state = opt_init(init_params)

def step(i, opt_state, opt_update):
    params = get_params(opt_state)
    g = grad(cost)(params)
    return opt_update(i, g, opt_state)

epochs = 200
loss_hist = []

for epoch in range(epochs):
    opt_state = step(epoch, opt_state, opt_update)
    params = get_params(opt_state)
    loss = cost(params)
    loss_hist.append(loss)
    progress = [epoch+1, loss]
    if (epoch % 50 == 49):
        print("Epoch: {:2f} | Loss: {:3f}".format(*jnp.asarray(progress)))

print("\nBest action: Rx = {}. Ry = {}. Rz = {}.".format(params[0], params[1], params[2]))   
output = circuit(params)
print("Quantum state output = {}.\n".format(output))    

plt.plot(loss_hist)
plt.ylabel("Loss")
plt.xlabel("epochs")
plt.show()