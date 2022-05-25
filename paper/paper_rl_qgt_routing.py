import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
from tqdm import trange

from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Operator
from qiskit.extensions import RXGate, RYGate, RZGate    
from qiskit.quantum_info.states.measures import state_fidelity

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

def output_state(dx,dy,dz):
    I = np.array([[1, 0],
                [0, 1]])
    X = np.array([[0, 1],
                [1, 0]])    
    I_f = np.kron(I, I)
    X_f = np.kron(X, X)
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

def checkear_nozero(rx,ry,rz):
    outputstate = output_state(rx,ry,rz)
    target_state = [0, np.sqrt(2)/2, np.sqrt(2)/2, 0]
    fidelity = state_fidelity(outputstate, target_state)
    return (np.abs(outputstate[0]) != 1), fidelity

def reward_qnet(rx, ry, rz, n3):    

    zero_check, fidelity = checkear_nozero(rx,ry,rz)
    if  zero_check == 0:
        return [-200, fidelity]

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

    return [-(t1 + temp), fidelity]

################################################## Network settings

class Bandit:
    def __init__(self, all_actions, epsilon_0=0.1, e_decay=1, step_size=0.1, sample_averages=False, gradient=False):
        self.epsilon_0 = epsilon_0
        self.e_decay = e_decay
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.gradient = gradient

        self.all_actions = all_actions
        self.indices = np.arange(len(self.all_actions))

    def reset(self):
        if self.gradient==True:
            self.q_estimation = np.zeros(len(self.indices))
        else:
            self.q_estimation = np.zeros(len(self.indices)) - 200
        self.epsilon = self.epsilon_0
        self.action_count = np.zeros(len(self.indices))
        self.average_reward = 0
        self.time = 0

    def act(self):
        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)
        self.epsilon *= self.e_decay
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)
        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    def step(self, action, n3):
        rotat = self.all_actions[action]
        [reward, fidelity] = reward_qnet(rotat[0],rotat[1],rotat[2], n3)
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            self.step_size = 1 / self.action_count[action]
        if self.gradient:
            one_hot = np.zeros(len(self.all_actions))
            one_hot[action] = 1
            baseline = self.average_reward
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward, fidelity

def simulate(bandits, all_actions, n3, time, runs, labels):
    rewards = np.zeros((len(bandits), runs, time))
    rewards_avg = np.zeros(rewards.shape)  
    perf_network = np.zeros((len(bandits), runs, time))
    perf_network_avg = np.zeros(perf_network.shape) 
    fidelities = np.zeros((len(bandits), runs, time))
    fidelities_avg = np.zeros(fidelities.shape)   
    action_bst = None
    reward_bst = -200
    for i, bandit in enumerate(bandits):
        print("---> AGENT : [{}].".format(labels[i]))
        for r in range(runs):   
            print("Iterations : {}/{}.".format(r+1, runs))
            bandit.reset()
            tipo = 0
            for t in trange(time):
                if t >= n3[1][tipo+1]:
                    tipo += 1                
                action = bandit.act()
                reward, fidelity = bandit.step(action, n3[0][tipo])
                if reward > reward_bst:
                    action_bst = action
                    reward_bst = reward
                rewards[i, r, t] = -reward
                rewards_avg[i, r, t] = np.mean(rewards[i,r,n3[1][tipo]:t+1])
                perf_network[i, r, t] =  n3[2][tipo] / rewards[i, r, t]
                perf_network_avg[i, r, t] = np.mean(perf_network[i,r,n3[1][tipo]:t+1])
                fidelities[i, r, t] = fidelity
                fidelities_avg[i, r, t] = np.mean(fidelities[i,r,n3[1][tipo]:t+1])

            rotat = all_actions[action_bst]
            output = np.array(output_state(rotat[0], rotat[1], rotat[2]))
            [rwd_bst, flt_bst] = reward_qnet(rotat[0],rotat[1],rotat[2], n3[0][tipo])
            prf_bst = -n3[2][tipo] / rwd_bst
            print("Best action: Rx = {}. Ry = {}. Rz = {}.".format(rotat[0], rotat[1], rotat[2]))
            print("Quantum state output = {}.".format(output))
            print("Performance_bst = {}. Fidelity_bst = {}.".format(prf_bst, flt_bst))
            print("Performance = {}. Fidelity = {}.\n".format(perf_network_avg[i,r,-1], fidelities_avg[i,r,-1]))

    mr = rewards.mean(axis=1)
    mr_avg = rewards_avg.mean(axis=1)
    mp = perf_network.mean(axis=1)
    mp_avg = perf_network_avg.mean(axis=1)
    mf = fidelities.mean(axis=1)
    mf_avg = fidelities_avg.mean(axis=1)
    return mr, mr_avg, mp, mp_avg, mf, mf_avg

def VQC():
    N_SIZE = 3
    angulos = np.arange(0, 2 * np.pi, 2 * np.pi / np.power(2, N_SIZE))
    all_actions = [(rx,ry,rz) for rx in angulos for ry in angulos for rz in angulos]
    time = 512
    runs = 10
    # maxima distancio 140  -> best time 153.338
    # maxima distancio 14   -> best time 37.4592
    # maxima distancia 0.14 -> best time 16.4807
    n3 = [[14],
          [0, time],
          [37.4592]]
    epsilon_0 = 0.99
    epsilon_f = 0.01
    e_decay = np.power(epsilon_f/epsilon_0, 1/time)    

    bandits = []
    bandits.append(Bandit(all_actions=all_actions))
    bandits.append(Bandit(all_actions=all_actions, sample_averages=True))
    bandits.append(Bandit(all_actions=all_actions, epsilon_0=epsilon_0, e_decay=e_decay))
    bandits.append(Bandit(all_actions=all_actions, epsilon_0=epsilon_0, e_decay=e_decay, sample_averages=True))
    bandits.append(Bandit(all_actions=all_actions, gradient=True))
    bandits.append(Bandit(all_actions=all_actions, sample_averages=True, gradient=True))
    labels = [
        'e = 0.1, a = 0.1',
        'e = 0.1, a = 1/n', 
        'e = decay, a = 0.1', 
        'e = decay, a = 1/n', 
        'gradient ascent, a = 0.1', 
        'gradient ascent, a = 1/n'
    ]
    rewards , rewards_avg, perf_network, perf_network_avg, fidelities, fidelities_avg = simulate(bandits, all_actions, n3, time, runs, labels)
    
    fig, axs = plt.subplots(2, 2, figsize=(30,30))
    for i in range(len(bandits)):
        axs[0,0].plot(rewards[i], label=labels[i])
        axs[1,0].plot(rewards_avg[i], label=labels[i])
        axs[0,1].plot(perf_network_avg[i], label=labels[i])
        axs[1,1].plot(fidelities_avg[i], label=labels[i])

    axs[0,0].set_title("Learning Quantum Strategies (Total time)")
    axs[1,0].set_title("Learning Quantum Strategies (Avg time)")
    axs[0,1].set_title("Learning Quantum Strategies (Avg performance)")
    axs[1,1].set_title("Learning Quantum Strategies (Fidelity)")
    axs[0,0].set_ylabel("Total Time")
    axs[1,0].set_ylabel("Avg Time")
    axs[0,1].set_ylabel("Mean Perf %")
    axs[1,1].set_ylabel("Mean Fidelity")
    axs[1,0].set_xlabel("Episodes")
    axs[1,1].set_xlabel("Episodes")
    axs[0,0].legend()
    plt.show()

if __name__ == '__main__':
    VQC()