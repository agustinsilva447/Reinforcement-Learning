import pickle                      
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator
from qiskit.extensions import RXGate, RYGate, RZGate          

##################################################
def generar_mapa(n1, n3):
    a = 0
    while (np.linalg.matrix_rank(a)!=n1):
        a = np.random.randint(n3, size=(n1,n1))
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
        # barrido
        dx = tipo[0]
        dy = tipo[1]
        dz = tipo[2]

    for q in range(n):
        circ.append(RXGate(dx),[q])
        circ.append(RYGate(dy),[q])
        circ.append(RZGate(dz),[q])    

    circ.append(J_dg, range(n))
    circ.measure(range(n), range(n))  
    return circ

def juego(lista, tipo):
    m = len(lista)
    if m > 0:
        for r in range(int(np.ceil(np.log2(m)))):
            ganadores = []            
            for j in range(int(np.ceil(m/2))):
                jug = 2 - int(m == j+int(np.ceil(m/2)))                        
                circ = crear_circuito(jug, tipo)
                backend = Aer.get_backend('qasm_simulator')
                measurement = execute(circ, backend=backend, shots=1).result().get_counts(circ)
                for k,i in enumerate(list(measurement.keys())[0]):
                    if i=='1':
                        ganadores.append(lista[2*j + k])                    
            lista = ganadores   
            m = len(lista)         
    return lista

def checkear_nozero(check):
    circ = crear_circuito(2, check)
    backend = Aer.get_backend('qasm_simulator')
    measurement = execute(circ, backend=backend, shots=1000).result().get_counts(circ)
    return ['00'] != list(measurement.keys())

def reward_qnet(rx, ry, rz):    

    if checkear_nozero([rx,ry,rz,1]) == 0:
        return 200

    n1 = 10                                                                                         # cantidad de ciudades
    n2 = 100                                                                                        # cantidad de paquetes
    n3 = 10                                                                                         # distancia máxima
    n4 = 2                                                                                          # cantidad de iteraciones
    p1 = [rx, ry, rz, 1]

    t = 0
    t1 = 0
    t2 = 0
    coste = 0
    dr = 0

    for p in range(n4):
        a = generar_mapa(n1, n3)                      # genero matriz
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
            t += 1 
            t1 += 1
            all_edges = [e for e in net1.edges]
            paquetes_ruta = paquetes_en_ruta(caminitos, all_edges[i], n2)
            if paquetes_ruta == []:
                t1 -= 1  
                t2 += 1  
                i += 1
            else:
                i = 0
                ganadores = juego(paquetes_ruta, p1)
                for x in range(len(ganadores)):
                    moves[ganadores[x]] = [-1,-2]
                    for y in caminitos[ganadores[x]]:
                        veces[np.where((np.array(all_edges2) == y).all(axis=1))[0][0]] += 1
                        tiemp += 2 * net2[y[0]][y[1]]['weight'] * veces[np.where((np.array(all_edges2) == y).all(axis=1))[0][0]] - 1
                        net1.remove_edges_from([y])
                        net2[y[0]][y[1]]['color'] = colores[envio]
                    envio += 1
                caminitos, flag = caminos(net1, moves)
        try:
            temp = tiemp/envio    #tiempo de envío por paquete 
        except ZeroDivisionError:
            temp = 2*n3            
        coste += temp   
        dr += (envio)/(n2)
    dr = (dr)/(n4)
    t = t / n4
    t1 = t1 / n4             # routing time
    t2 = t2 / n4
    coste = coste / n4       #traveling time    

    return (t1 + coste)

##################################################

epsilon = 0.99              # randomness
EPS_DECAY = 0.999           
HM_EPISODES = 500
NET_STATES = 1              # 1 para red solo congestionada, 2 para red congestionada o no
N_SIZE = 3
angulos = np.arange(0, 2 * np.pi, 2 * np.pi / np.power(2, N_SIZE))
all_actions = [(rx,ry,rz) for rx in angulos for ry in angulos for rz in angulos] 
start_q_table = None        # if we have a pickled Q table, we'll put the filename of it here.

if start_q_table is None:
    q_table = {}
    for rx in angulos:
        for ry in angulos:
            for rz in angulos:
                q_table[(rx,ry,rz)] = [-np.random.uniform(100, 200) for i in range(NET_STATES)]     
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []
for episode in range(HM_EPISODES):
    episode_reward = 0
 
    #obs = acá habría que cargar el estado pero por ahora es uno solo
    if np.random.random() > epsilon:
        max_value = max(q_table.values())
        action = [k for k, v in q_table.items() if v == max_value][0]
    else:
        action = random.choice(all_actions)
    reward = -reward_qnet(action[0], action[1], action[2])
    q_table[action] = [reward]
    
    episode_rewards.append(reward)
    epsilon *= EPS_DECAY      
    
    print("Episode: {}. Reward: {}. Action: {}.".format(episode, reward, action))

max_value = max(q_table.values())
action = [k for k, v in q_table.items() if v == max_value][0]
print("Best action: Rx = {}. Ry = {}. Rz = {}.".format(action[0], action[1], action[2]))    
print("Total time = {} secods.".format(reward_qnet(action[0], action[1], action[2])))

