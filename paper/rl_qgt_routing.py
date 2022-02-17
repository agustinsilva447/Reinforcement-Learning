import copy
import time
import pickle                      
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator
from qiskit.extensions import RXGate, RYGate, RZGate          

################################################## Quantum Game
def generar_mapa(n1, n3):
    a = 0
    while (np.linalg.matrix_rank(a)!=n1):
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

def state_out_new_p(x,y,z):
    # all these were calculated with sympy in order to speed computing time
    sqrt_2 = np.sqrt(2)
    cos_x = np.cos(x/2)
    sin_x = np.sin(x/2)
    cos_y = np.cos(y/2)
    sin_y = np.sin(y/2)
    exp_z = np.exp(1j*z)
    exp_m = np.exp(-1j*z)

    s_0 = sqrt_2/2*(-sqrt_2/2*1j*exp_z*sin_y**2 + sqrt_2/2*exp_m*cos_y**2)*cos_x**2 - sqrt_2/2*(-sqrt_2/2*1j*exp_z*cos_y**2 + sqrt_2/2*exp_m*sin_y**2)*sin_x**2 - sqrt_2*1j*(-sqrt_2/2*1j*exp_z*sin_y*cos_y - sqrt_2/2*exp_m*sin_y*cos_y)*sin_x*cos_x + sqrt_2/2*1j*(-(-sqrt_2/2*1j*exp_z*sin_y**2 + sqrt_2/2*exp_m*cos_y**2)*sin_x**2 + (-sqrt_2/2*1j*exp_z*cos_y**2 + sqrt_2/2*exp_m*sin_y**2)*cos_x**2 - 2*1j*(-sqrt_2/2*1j*exp_z*sin_y*cos_y - sqrt_2/2*exp_m*sin_y*cos_y)*sin_x*cos_x)
    s_1 = -sqrt_2/2*1j*(sqrt_2/2*1j*sin_y**2 + sqrt_2/2*cos_y**2)*sin_x*cos_x - sqrt_2/2*(-sqrt_2/2*sin_y*cos_y + sqrt_2/2*1j*sin_y*cos_y)*sin_x**2 + sqrt_2/2*(sqrt_2/2*sin_y*cos_y - sqrt_2/2*1j*sin_y*cos_y)*cos_x**2 - sqrt_2/2*1j*(-sqrt_2/2*sin_y**2 - sqrt_2/2*1j*cos_y**2)*sin_x*cos_x + sqrt_2/2*1j*(-1j*(sqrt_2/2*1j*sin_y**2 + sqrt_2/2*cos_y**2)*sin_x*cos_x + (-sqrt_2/2*sin_y*cos_y + sqrt_2/2*1j*sin_y*cos_y)*cos_x**2 - (sqrt_2/2*sin_y*cos_y - sqrt_2/2*1j*sin_y*cos_y)*sin_x**2 - 1j*(-sqrt_2/2*sin_y**2 - sqrt_2/2*1j*cos_y**2)*sin_x*cos_x)
    s_2 = -sqrt_2/2*1j*(sqrt_2/2*1j*sin_y**2 + sqrt_2/2*cos_y**2)*sin_x*cos_x - sqrt_2/2*(-sqrt_2/2*sin_y*cos_y + sqrt_2/2*1j*sin_y*cos_y)*sin_x**2 + sqrt_2/2*(sqrt_2/2*sin_y*cos_y - sqrt_2/2*1j*sin_y*cos_y)*cos_x**2 - sqrt_2/2*1j*(-sqrt_2/2*sin_y**2 - sqrt_2/2*1j*cos_y**2)*sin_x*cos_x + sqrt_2/2*1j*(-1j*(sqrt_2/2*1j*sin_y**2 + sqrt_2/2*cos_y**2)*sin_x*cos_x + (-sqrt_2/2*sin_y*cos_y + sqrt_2/2*1j*sin_y*cos_y)*cos_x**2 - (sqrt_2/2*sin_y*cos_y - sqrt_2/2*1j*sin_y*cos_y)*sin_x**2 - 1j*(-sqrt_2/2*sin_y**2 - sqrt_2/2*1j*cos_y**2)*sin_x*cos_x)
    s_3 = sqrt_2/2*(sqrt_2/2*exp_z*sin_y**2 - sqrt_2/2*1j*exp_m*cos_y**2)*cos_x**2 - sqrt_2/2*(sqrt_2/2*exp_z*cos_y**2 - sqrt_2/2*1j*exp_m*sin_y**2)*sin_x**2 - sqrt_2*1j*(sqrt_2/2*exp_z*sin_y*cos_y + sqrt_2/2*1j*exp_m*sin_y*cos_y)*sin_x*cos_x + sqrt_2/2*1j*(-(sqrt_2/2*exp_z*sin_y**2 - sqrt_2/2*1j*exp_m*cos_y**2)*sin_x**2 + (sqrt_2/2*exp_z*cos_y**2 - sqrt_2/2*1j*exp_m*sin_y**2)*cos_x**2 - 2*1j*(sqrt_2/2*exp_z*sin_y*cos_y + sqrt_2/2*1j*exp_m*sin_y*cos_y)*sin_x*cos_x)
    p_new = np.round(np.array([s_0, s_1, s_2, s_3]).reshape(1,4),5)
    p_new = np.abs(p_new)**2
    return (p_new)/(np.sum(p_new))

def state_out_new(n, tipo):
    if n==1:
        a = {'1': 1}
        return a
    elif n == 2:
        a0 = {'00': 1}
        a1 = {'01': 1}
        a2 = {'10': 1}
        a3 = {'11': 1}
        x = [a0, a1, a2, a3]
        p_new = state_out_new_p(tipo[0], tipo[1], tipo[2])[0]
        return np.random.choice(x, p = [p_new[0], p_new[1], p_new[2], p_new[3]])

def juego(lista, tipo):
    m = len(lista)
    if m > 0:
        for r in range(int(np.ceil(np.log2(m)))):
            ganadores = []            
            for j in range(int(np.ceil(m/2))):
                jug = 2 - int(m == j+int(np.ceil(m/2)))                        
                measurement = state_out_new(jug, tipo)
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

def output_state(dx,dy,dz):
    I_f = np.array([[1, 0],
                [0, 1]]) 
    I = np.array([[1, 0],
                [0, 1]])
    X_f = np.array([[0, 1],
                [1, 0]]) 
    X = np.array([[0, 1],
                [1, 0]])    
    for q in range(1):
        I_f = np.kron(I_f, I)
        X_f = np.kron(X_f, X)
    J = Operator(1 / np.sqrt(2) * (I_f + 1j * X_f))    
    J_dg = J.adjoint()
    circ = QuantumCircuit(2,2)
    circ.append(J, range(2))

    for q in range(2):
        circ.append(RXGate(dx),[q])
        circ.append(RYGate(dy),[q])
        circ.append(RZGate(dz),[q])    

    circ.append(J_dg, range(2))
    backend = Aer.get_backend('statevector_simulator')
    job = backend.run(circ)
    result = job.result()
    outputstate = result.get_statevector(circ, decimals=5)
    return outputstate

def reward_qnet(rx, ry, rz, n3):    

    if checkear_nozero([rx,ry,rz,1]) == 0:
        return 200

    n1 = 10                                       # cantidad de ciudades
    n2 = 100                                      # cantidad de paquetes
    p1 = [rx, ry, rz, 1]

    a = generar_mapa(n1, n3)                      # genero matriz
    net1, edge_weights_list = generar_red(a)      # genero red
    net2, edge_weights_list = generar_red(a)      # genero copia de red
    moves, colores = generar_paquetes(n1,n2)      # genero paquetes
    caminitos, flag = caminos(net1, moves)        # caminos óptimos
    all_edges2 = [e for e in net2.edges]
    veces = np.zeros(len(all_edges2))

    i = 0
    t1 = 0
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
            ganadores = juego(paquetes_ruta, p1)
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

    return (t1 + temp)

################################################## Network settings

HM_EPISODES = 512
SHOW_EVERY = 1
N_SIZE = 3
n3 = [[0.14],                    # distancias máximas
      [0, HM_EPISODES]]

"""
HM_EPISODES = 2 * 4092 + 1
SHOW_EVERY = 128
N_SIZE = 4
n3 = [[0.14, 14],                    # distancias máximas
      [0, 4092, HM_EPISODES]]
"""

final_t = []
final_m = []
angulos = np.arange(0, 2 * np.pi, 2 * np.pi / np.power(2, N_SIZE))
all_actions = [(rx,ry,rz) for rx in angulos for ry in angulos for rz in angulos] 

################################################## Epsilon and Alfa

start_q_table = None        # if we have a pickled Q table, we'll put the filename of it here.
epsilon =   [[0.99, 0.991, '1/n']]    # epsilon variable y alfa variable
             #[0.1 , 1 , '1/n'],      # epsilon constante y alfa variable
             #[0.1 , 1 , 0.1],        # epsilon constante y alfa constante
             #[0.99 , 0.991 , 0.1]]   # epsilon variable y alfa constante

for it in range(len(epsilon)):
    print("ITERATION: {}.".format(it+1))
    if start_q_table is None:
        q_table = {}
        n_actions = {}
        for rx in angulos:
            for ry in angulos:
                for rz in angulos:
                    q_table[(rx,ry,rz)] = -np.random.uniform(100, 200)    
                    n_actions[(rx,ry,rz)] = 0     
    else:
        with open(start_q_table, "rb") as f:
            q_table = pickle.load(f)

    type = 0
    episode_reward = []
    episode_rewards = []
    for episode in range(HM_EPISODES):
        
        if np.random.random() > epsilon[it][0]:
            max_value = max(q_table.values())
            action = [k for k, v in q_table.items() if v == max_value][0]
        else:
            action = random.choice(all_actions)

        if episode > (n3[1][type + 1]):
            type += 1

        reward = -reward_qnet(action[0], action[1], action[2], n3[0][type])
        episode_reward.append(reward)
        episode_rewards.append(np.mean(episode_reward[n3[1][type]::]))

        n_actions[action] += 1
        if epsilon[it][2] == '1/n':
            alfa = (1/n_actions[action])
        else:
            alfa = epsilon[it][2]
        q_table[action] = q_table[action] + alfa * (reward - q_table[action])       
        epsilon[it][0] *= epsilon[it][1]       

        if episode % SHOW_EVERY == 0:
            print("---> Episode: {}. Epsilon: {:.4f}. Q-table: {:.4f}. Reward: {:.4f}. Action: {}.".format(np.round(episode,4), np.round(epsilon[it][0],4), np.round(q_table[action], 4), np.round(reward,4), np.round(action,4)))            

    max_value = max(q_table.values())
    action = [k for k, v in q_table.items() if v == max_value][0]
    output = output_state(action[0], action[1], action[2])
    print("\nBest action: Rx = {}. Ry = {}. Rz = {}.".format(action[0], action[1], action[2]))    
    print("Total time = {} secods.".format(-q_table[action]))
    print("Quantum state output = {}.\n".format(output))

    episode_reward = np.negative(np.array(episode_reward))
    episode_rewards = np.negative(np.array(episode_rewards))
    final_t.append(episode_reward)
    final_m.append(episode_rewards)

################################################## Stochastic Gradient Ascent

"""alfa = 0.1
start_h_table = None        # if we have a pickled Q table, we'll put the filename of it here.

if start_h_table is None:
    h_table = {}
    policy_p = {}
    for rx in angulos:
        for ry in angulos:
            for rz in angulos:
                h_table[(rx,ry,rz)] = 0     
                policy_p[(rx,ry,rz)] = 0
else:
    with open(start_h_table, "rb") as f:
        h_table = pickle.load(f)

type = 0
episode_reward = []
episode_rewards = []
for episode in range(HM_EPISODES):
    
    result = h_table.values()
    data = list(result)
    numpyArray_h = np.array(data)
    denominador = np.sum(np.exp(numpyArray_h))
    
    for rx in angulos:
        for ry in angulos:
            for rz in angulos:    
                policy_p[(rx,ry,rz)] = (np.exp(h_table[(rx,ry,rz)]))/(denominador)
    
    result = policy_p.values()
    data_p = list(result)       
    action = random.choices(all_actions, weights = data_p)[0]

    if episode > (n3[1][type + 1]):
        type += 1

    reward = -reward_qnet(action[0], action[1], action[2], n3[0][type])
    episode_reward.append(reward)
    episode_rewards.append(np.mean(episode_reward[n3[1][type]::]))
    reward_mean = np.mean(episode_reward)

    for rx in angulos:
        for ry in angulos:
            for rz in angulos: 
                if action == (rx,ry,rz):
                    h_table[(rx,ry,rz)] = h_table[(rx,ry,rz)] + alfa * (reward - reward_mean) * (1 - policy_p[(rx,ry,rz)]) 
                else:
                    h_table[(rx,ry,rz)] = h_table[(rx,ry,rz)] - alfa * (reward - reward_mean) * (policy_p[(rx,ry,rz)])

    if episode % SHOW_EVERY == 0:
        print("---> Episode: {}. Action: {}. H-table: {:.4f}. Policy: {:.4f}. Reward: {:.4f}.".format(np.round(episode,4), np.round(action,4), np.round(h_table[action], 4), np.round(policy_p[action],4), np.round(reward,4)))

max_value = max(h_table.values())
action = [k for k, v in h_table.items() if v == max_value][0]
output = output_state(action[0], action[1], action[2])
print("\nBest action: Rx = {}. Ry = {}. Rz = {}.".format(action[0], action[1], action[2]))    
print("Total time = {} secods.".format(-episode_rewards[-1]))
print("Quantum state output = {}.\n".format(output))

episode_reward = np.negative(np.array(episode_reward))
episode_rewards = np.negative(np.array(episode_rewards))
final_t.append(episode_reward)
final_m.append(episode_rewards)"""

################################################## Algorithms Comparison

fig, axs = plt.subplots(2, 1) #,figsize=(30,20))
axs[0].set_title("Learning Rx, Ry and Rz for the congestion mitigation problem.")
axs[0].plot(final_t[0], label = "Epsilon variable y Alfa variable")
#axs[0].plot(final_t[1], label = "Epsilon constante y Alfa variable")
#axs[0].plot(final_t[2], label = "Epsilon constante y Alfa constante")
#axs[0].plot(final_t[3], label = "Epsilon variable y Alfa constante")
#axs[0].plot(final_t[4], label = "Stochastic Gradient Ascent")
axs[0].set_ylabel("Total Time")
axs[0].legend(loc='upper right')
axs[1].set_title("Learning Rx, Ry and Rz for the congestion mitigation problem [average].")
axs[1].plot(final_m[0], label = "Epsilon variable y Alfa variable")
#axs[1].plot(final_m[1], label = "Epsilon constante y Alfa variable")
#axs[1].plot(final_m[2], label = "Epsilon constante y Alfa constante")
#axs[1].plot(final_m[3], label = "Epsilon variable y Alfa constante")
#axs[1].plot(final_m[4], label = "Stochastic Gradient Ascent")
axs[1].set_ylabel("Total Time")
axs[1].set_xlabel("Episodes")
axs[1].legend(loc='upper right')
plt.show()

"""with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)"""