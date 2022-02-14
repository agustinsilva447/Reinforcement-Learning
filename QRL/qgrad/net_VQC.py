import numpy as np
import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax import grad

import matplotlib.pyplot as plt
from functools import reduce

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
    layer0 = jnp.array([[1], [0], [0], [0]])
    layer1 = J()
    layer2 = jnp.kron(rx(thetax), rx(thetax))
    layer3 = jnp.kron(ry(thetay), ry(thetay))
    layer4 = jnp.kron(rz(thetaz), rz(thetaz))
    layer5 = J_dag()
    layers = [layer5, layer4, layer3, layer2, layer1, layer0]
    q_state_out = reduce(lambda  x, y : jnp.dot(x, y), layers)
    return q_state_out

def cost(params):
    state = circuit(params)
    op_qs = jnp.array([[0], [0.707], [0.707], [0]])
    fid = jnp.abs(jnp.dot(jnp.transpose(jnp.conjugate(op_qs)), state)) ** 2
    return -jnp.real(fid)[0][0]

# fixed random parameter initialization
init_params = [2 * jnp.pi * np.random.rand(), 2 * jnp.pi * np.random.rand(), 2 * jnp.pi * np.random.rand()]
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)
opt_state = opt_init(init_params)

def step(i, opt_state, opt_update):
    params = get_params(opt_state)
    g = grad(cost)(params)
    return opt_update(i, g, opt_state)

epoch = 0
epoch_max = 250

loss_hist = []
loss = cost(init_params)

while ((loss > -0.999) and (epoch_max > epoch)): 
    opt_state = step(epoch, opt_state, opt_update)
    params = get_params(opt_state)
    loss = cost(params)
    loss_hist.append(loss)
    progress = [epoch+1, loss]
    if (epoch % 50 == 49):
        print("Epoch: {} | Loss: {}".format(*jnp.asarray(progress)))
    epoch += 1
print("Epoch: {} | Loss: {}".format(*jnp.asarray(progress)))

print("\nBest action: [Rx, Ry, Rz] = [{}, {}, {}]".format(params[0], params[1], params[2]))   
d = (np.array(params))/(np.pi)
print("Best action: [Rx, Ry, Rz] = [{:.2f}*π, {:.2f}*π, {:.2f}*π]\n".format(d[0], d[1], d[2])) 

output = circuit(params)
print("Quantum state output = {}.\n".format(output))    

plt.plot(loss_hist)
plt.ylabel("Loss")
plt.xlabel("epochs")
plt.show()