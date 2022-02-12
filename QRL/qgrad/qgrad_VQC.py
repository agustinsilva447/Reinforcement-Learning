import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad
from jax.example_libraries import optimizers

from functools import reduce
from qgrad.qgrad_qutip import basis, expect, sigmaz


def rx(phi):
    return jnp.array([[jnp.cos(phi / 2), -1j * jnp.sin(phi / 2)],
                     [-1j * jnp.sin(phi / 2), jnp.cos(phi / 2)]])

def ry(phi):
    return jnp.array([[jnp.cos(phi / 2), -jnp.sin(phi / 2)],
                     [jnp.sin(phi / 2), jnp.cos(phi / 2)]])

def rz(phi):
    return jnp.array([[jnp.exp(-1j * phi / 2), 0],
                     [0, jnp.exp(1j * phi / 2)]])

def cnot():
    return jnp.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]],)

def circuit(params):
    thetax, thetay, thetaz = params
    layer0 = jnp.kron(basis(2, 0), basis(2, 0))
    layer1 = jnp.kron(ry(jnp.pi / 4), ry(jnp.pi / 4))
    layer2 = jnp.kron(rx(thetax), jnp.eye(2))
    layer3 = jnp.kron(ry(thetay), rz(thetaz))
    layers = [layer1, cnot(), layer2, cnot(), layer3]
    unitary = reduce(lambda  x, y : jnp.dot(x, y), layers)
    return jnp.dot(unitary, layer0)                     

# pauli Z on the first qubit
op = jnp.kron(sigmaz(), jnp.eye(2))

def cost(params, op):
    state = circuit(params)
    return jnp.real(expect(op, state))

# fixed random parameter initialization
init_params = [0., 0., 0.]
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)
opt_state = opt_init(init_params)

def step(i, opt_state, opt_update):
    params = get_params(opt_state)
    g = grad(cost)(params, op)
    return opt_update(i, g, opt_state)

epochs = 400
loss_hist = []

for epoch in range(epochs):
    opt_state = step(epoch, opt_state, opt_update)
    params = get_params(opt_state)
    loss = cost(params, op)
    loss_hist.append(loss)
    progress = [epoch+1, loss]
    if (epoch % 50 == 49):
        print("Epoch: {:2f} | Loss: {:3f}".format(*jnp.asarray(progress)))

plt.plot(loss_hist)
plt.ylabel("Loss")
plt.xlabel("epochs")
plt.show()