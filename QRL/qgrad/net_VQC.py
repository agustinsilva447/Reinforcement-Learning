import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator
from qiskit.extensions import RXGate, RYGate, RZGate

import jax.numpy as jnp
from jax import grad
from jax.example_libraries import optimizers
from functools import reduce
from qgrad.qgrad_qutip import basis, expect, sigmaz

##################################################
def generar_mapa(n1, n3):
    a = 0
    while (np.linalg.matrix_rank(a)!=n1):
        #a = np.random.randint(n3, size=(n1,n1))
        a = n3 * np.random.rand(n1,n1)
        np.fill_diagonal(a,0)
        a = np.tril(a) + np.tril(a, -1).T
    return a

def generar_red(a):
    net1 = nx.from_numpy_matrix(copy.deepcopy(a))
    for e in net1.edges():
        net1[e[0]][e[1]]['color'] = 'black'
    edge_weights_list = [net1[e[0]][e[1]]['weight'] for e in net1.edges()]
    return net1, edge_weights_list

def generar_paquetes(n1,n2):
    moves = np.zeros([n2, 2])    
    rng = np.random.default_rng()
    for i in range(n2):
        moves[i,:] = rng.choice(n1, size=2, replace=False)    
    colores = []
    for i in range(n2):
        color = np.base_repr(np.random.choice(16777215), base=16)
        colores.append('#{:0>6}'.format(color))    
    return moves, colores

def caminos(net1, moves):
    caminitos = []
    i = 0
    for j in range(len(moves)):
        cam = []
        try:
            p = nx.dijkstra_path(net1,int(moves[j,0]),int(moves[j,1]))
            for e in range(len(p)-1):
                cam.append(tuple(sorted((p[e], p[e+1]))))    
        except:
            i += 1
            if i == len(moves):
                return caminitos, True        
        caminitos.append(cam)
    return caminitos, False

def paquetes_en_ruta(camin, ruta, n2):
    lista = []
    for i in range(n2):
        if ruta in camin[i]:
            lista.append(i)
    return lista

def juego(lista, probas):
    m = len(lista)
    if m > 0:
        for r in range(int(np.ceil(np.log2(m)))):
            ganadores = []            
            for j in range(int(np.ceil(m/2))):
                jug = 2 - int(m == j+int(np.ceil(m/2)))                        
                measurement = opciones_cuan(probas)
                for k,i in enumerate(list(measurement.keys())[0]):
                    if i=='1':
                        ganadores.append(lista[2*j + k])                    
            lista = ganadores   
            m = len(lista)         
    return lista    

def opciones_cuan(probas):
    a0 = {'00': 1}
    a1 = {'01': 1}
    a2 = {'10': 1}
    a3 = {'11': 1}
    x = [a0, a1, a2, a3]    
    pp = [probas[0], probas[1], probas[2], probas[3]]
    print(probas, "\n")
    print(pp, "\n")
    return np.random.choice(x, p = pp)

def crear_circuito(n, tipo):
    I_f = np.array([[1, 0],
                  [0, 1]]) 
    I = np.array([[1, 0],
                  [0, 1]])
    X_f = np.array([[0, 1],
                  [1, 0]]) 
    X = np.array([[0, 1],
                  [1, 0]])    
    for q in range(n-1):
        I_f = np.kron(I_f, I)
        X_f = np.kron(X_f, X)
    J = Operator(1 / np.sqrt(2) * (I_f + 1j * X_f))    
    J_dg = J.adjoint()
    circ = QuantumCircuit(n,n)
    circ.append(J, range(n))
    if n==1:
        dx = np.pi
        dy = 0
        dz = 0
    elif n==2:    
        # Pareto, Nash y Mixta
        #dx = np.pi/2
        #dy = np.pi/4
        #dz = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2])

        # barrido
        dx = tipo[0]
        dy = tipo[1]
        dz = tipo[2]

    """
    elif n==4:
        dx = np.pi/2
        dy = 3 * np.pi/8
        dz = 3 * np.pi/4    
    """

    for q in range(n):
        circ.append(RXGate(dx),[q])
        circ.append(RYGate(dy),[q])
        circ.append(RZGate(dz),[q])    

    circ.append(J_dg, range(n))
    circ.measure(range(n), range(n))  
    return circ    

def checkear_nozero(check):
    circ = crear_circuito(2, check)
    backend = Aer.get_backend('qasm_simulator')
    measurement = execute(circ, backend=backend, shots=1000).result().get_counts(circ)
    return ['00'] != list(measurement.keys())

def reward(probas):
    n1 = 10             # cantidad de ciudades
    n2_array = [100]    # cantidad de paquetes
    n3 = 14             # distancia máxima
    n4 = 10             # cantidad de iteraciones 

    for cant,n2 in enumerate(n2_array):    
        t1 = 0
        coste = 0

        for p in range(n4):
            a = generar_mapa(n1,n3)                       # genero matriz
            net1, edge_weights_list = generar_red(a)      # genero red
            net2, edge_weights_list = generar_red(a)      # genero copia de red
            moves, colores = generar_paquetes(n1,n2)      # genero paquetes
            caminitos, flag = caminos(net1, moves)        # caminos óptimos
            all_edges2 = [e for e in net2.edges]
            veces = np.zeros(len(all_edges2))
            i = 0
            tiemp = 0
            envio = 0
            while not flag:
                t1 += 1
                all_edges = [e for e in net1.edges]
                paquetes_ruta = paquetes_en_ruta(caminitos, all_edges[i], n2)
                if paquetes_ruta == []:
                    t1 -= 1  
                    i += 1
                else:
                    i = 0
                    ganadores = juego(paquetes_ruta, probas)
                    for x in range(len(ganadores)):
                        moves[ganadores[x]] = [-1,-2]
                        for y in caminitos[ganadores[x]]:
                            veces[np.where((np.array(all_edges2) == y).all(axis=1))[0][0]] += 1
                            tiemp += (2 * veces[np.where((np.array(all_edges2) == y).all(axis=1))[0][0]] - 1) * net2[y[0]][y[1]]['weight']
                            net1.remove_edges_from([y])
                            net2[y[0]][y[1]]['color'] = colores[envio]
                        envio += 1
                    caminitos, flag = caminos(net1, moves)
            try:
                temp = tiemp/envio    #tiempo de envío por paquete 
            except ZeroDivisionError:
                temp = 2*n3            
            coste += temp   
        t1 = t1 / n4
        coste = coste / n4
        print("{:0>3} - Traveling time = {}. Routing Time = {}\n".format(cant+1, coste, t1))

    return -(t1+coste)

##################################################

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
    for q in range(1):
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

##################################################

def cost(params):
    op = jnp.kron(sigmaz(), jnp.eye(2))
    state = circuit(params)
    probas = jnp.power(jnp.abs(state),2)
    return jnp.real(reward(probas))

def step(i, opt_state, opt_update):
    params = get_params(opt_state)
    g = grad(cost)(params)
    return opt_update(i, g, opt_state) 

init_params = [-jnp.pi/2, 0., 0.]
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)
opt_state = opt_init(init_params)

epochs = 10
loss_hist = []

for epoch in range(epochs):
    opt_state = step(epoch, opt_state, opt_update)
    params = get_params(opt_state)
    loss = cost(params)
    loss_hist.append(loss)
    progress = [epoch+1, loss]
    if (epoch % 1 == 0):
        print("Epoch: {:2f} | Loss: {:3f}".format(*jnp.asarray(progress)))

print(params)
plt.plot(loss_hist)
plt.ylabel("Loss")
plt.xlabel("epochs")
plt.show()