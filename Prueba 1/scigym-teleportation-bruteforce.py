import matplotlib.pyplot as plt
import numpy as np
import scigym
from qiskit import Aer
from qiskit import execute
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram, plot_state_city

j = 0; reward = 0
while (reward == 0) and (j<1000) :
    i = 0; j += 1
    env = scigym.make('teleportation-v0')  
    observation = env.reset()  
    done = False
    actions_int = []
    num_actions = env.action_space.n; available = {}
    available['available_actions'] = range(num_actions)
    while done==False:  
        i += 1
        action = np.random.choice(available['available_actions'])
        actions_int.append(action)
        (observation, reward, done, available) = env.step(action)
        print("{}, {} --> Reward: {}. Done: {}. Available actions: {}.".format(j, i, reward, done, available['available_actions']), end=" ")      
        if bool(available['available_actions']):
            print("Action: {}.".format(action))
        else:
            print("No more available actions.")
    if reward == 0:
        print("Target not met.")

if reward == 1:
    actions_gate = []
    circuit = QuantumCircuit(3,3)

    #setting input state
    circuit.x(0) 
    circuit.barrier() 
    #creating bell state
    circuit.h(1)
    circuit.cx(1,2)
    circuit.barrier() 

    for action in actions_int:
        if action == 0:
            actions_gate.append('H_0')
            circuit.h(0)
            circuit.barrier() 
        elif action == 1:
            actions_gate.append('T_0')
            circuit.t(0)
            circuit.barrier() 
        elif action == 2:
            actions_gate.append('H_1')
            circuit.h(1)
            circuit.barrier() 
        elif action == 3:
            actions_gate.append('T_1')
            circuit.t(1)
            circuit.barrier() 
        elif action == 4:
            actions_gate.append('H_2')
            circuit.h(2)
            circuit.barrier() 
        elif action == 5:
            actions_gate.append('T_2')
            circuit.t(2)
            circuit.barrier() 
        elif action == 6:
            actions_gate.append('CNOT_01')
            circuit.cx(0,1)
            circuit.barrier() 
        elif action == 7:
            actions_gate.append('MEASURE_0')
            #circuit.measure(0, 0) 
            #circuit.barrier() 
        elif action == 8:
            actions_gate.append('MEASURE_1')
            #circuit.measure(1, 1) 
            #circuit.barrier() 
        elif action == 9:
            actions_gate.append('MEASURE_2')
            #circuit.measure(2, 2) 
            #circuit.barrier() 

    circuit.cz(0, 2)
    circuit.cx(1, 2)
    circuit.barrier() 
    circuit.measure([0, 1, 2], [0, 1, 2]) 
    circuit.barrier() 

    print("\nNumber of actions: {}. Actions (gates): {}.\nQuantum Circuit:".format(i, actions_gate))
    print(circuit)

    backend = Aer.get_backend('statevector_simulator')
    result = execute(circuit, backend=backend).result()    
    statevector = result.get_statevector(circuit)
    print(statevector)
    fig = plot_state_city(statevector)
    fig.savefig("/home/agustinsilva447/Github/Reinforcement-Learning/Prueba 1/state.png")

    backend = Aer.get_backend('qasm_simulator')
    result = execute(circuit, backend=backend, shots=1000).result()
    counts = result.get_counts(circuit)
    print(counts)
    fig = plot_histogram(counts) 
    fig.savefig("/home/agustinsilva447/Github/Reinforcement-Learning/Prueba 1/counts.png")