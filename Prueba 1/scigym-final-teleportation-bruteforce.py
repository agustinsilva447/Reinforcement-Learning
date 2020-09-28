import matplotlib.pyplot as plt
import numpy as np
import scigym
from qiskit import Aer, execute, QuantumCircuit
from qiskit.visualization import plot_histogram, plot_state_city

def print_state(j, i, reward, done, available, action):
    if ((j%500)==0):
        print("{}, {} --> Reward: {}. Done: {}. Available actions: {}.".format(j, i, reward, done, available), end=" ")      
        if bool(available):
            print("Action: {}.".format(action))
        else:
            print("No more available actions.\n")

def pre_circuit(circuit):
    #setting input state
    circuit.x(0) 
    input = 1
    circuit.barrier() 
    #creating bell state
    circuit.h(1)
    circuit.cx(1,2)
    circuit.barrier()
    return circuit, input

def teleportation_circuit(observation, circuit):
    actions_gate = []
    for action in observation:
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
        elif (action == 10) or (action == 11):
            actions_gate.append('MEASURE_0')
            circuit.measure(0, 0) 
            circuit.barrier() 
        elif (action == 12) or (action == 13):
            actions_gate.append('MEASURE_1')
            circuit.measure(1, 1) 
            circuit.barrier() 
        elif (action == 14) or (action == 15):
            actions_gate.append('MEASURE_2')
            circuit.measure(2, 2) 
            circuit.barrier() 
    return circuit, actions_gate

def circuit_measurement(observation, circuit):
    outcome = [] 
    if (14 not in observation) or (15 not in observation) :
        circuit.measure([2], [2]) 
        circuit.barrier() 
    if 13 in observation:
        outcome.append(1)
    if 12 in observation:  
        outcome.append(0)
    if 11 in observation:
        outcome.append(1)
    if 10 in observation:
        outcome.append(0) 
    return circuit, outcome

def circuit_statevector_solver(circuit):    
    backend = Aer.get_backend('statevector_simulator')
    result = execute(circuit, backend=backend).result()    
    statevector = result.get_statevector(circuit)
    print("State vector: {}".format(statevector))
    fig = plot_state_city(statevector)
    fig.savefig("/home/agustinsilva447/Github/Reinforcement-Learning/Prueba 1/state.png")
    return statevector

def circuit_counts_solver(circuit):
    backend = Aer.get_backend('qasm_simulator')
    result = execute(circuit, backend=backend, shots=1000).result()
    counts = result.get_counts(circuit)
    print("Counts: {}".format(counts))
    fig = plot_histogram(counts) 
    fig.savefig("/home/agustinsilva447/Github/Reinforcement-Learning/Prueba 1/counts.png")
    return counts

j = 0; reward = 0
while (reward == 0):
    i = 0; j += 1
    env = scigym.make('teleportation-v0')  
    observation = env.reset()  
    done = False
    num_actions = env.action_space.n; available = {}
    available['available_actions'] = range(num_actions)
    while done==False:  
        i += 1
        action = np.random.choice(available['available_actions'])
        (observation, reward, done, available) = env.step(action)
        print_state(j, i, reward, done, available['available_actions'], action)

circuit = QuantumCircuit(3,3)
circuit, input = pre_circuit(circuit)
circuit, actions_gate = teleportation_circuit(observation, circuit)
circuit, outcome = circuit_measurement(observation, circuit)

print("{}: Number of actions: {}. Actions (gates): {}.".format(j, i, actions_gate))
print("Observations: {}.".format(observation))     
print("Quantum circuit:\n{}".format(circuit))
print("[qubit1, qubit0] = {}".format(outcome))
statevector = circuit_statevector_solver(circuit)
counts = circuit_counts_solver(circuit)

for out in counts.keys():
    if (int(out[2])==outcome[1]) and (int(out[1])==outcome[0]):
        output = int(out[0])
        if input == output:
            print("SUCCESSFUL communication!. Input qubit0 = {}. Output qubit2 = {}.".format(input, output))
        else :
            print("FAILED communication!. Input qubit0 = {}. Output qubit2 = {}.".format(input, output))