import matplotlib.pyplot as plt
import numpy as np
import scigym
from qiskit import Aer
from qiskit import execute
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

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

    circuit.x(0) 
    circuit.barrier() 
    circuit.h(1)
    circuit.cx(1,2)
    circuit.barrier() 

    for action in actions_int:
        if action == 0:
            actions_gate.append('H_0')
            circuit.h(0)
        elif action == 1:
            actions_gate.append('T_0')
            circuit.t(0)
        elif action == 2:
            actions_gate.append('H_1')
            circuit.h(1)
        elif action == 3:
            actions_gate.append('T_1')
            circuit.t(1)
        elif action == 4:
            actions_gate.append('H_2')
            circuit.h(2)
        elif action == 5:
            actions_gate.append('T_2')
            circuit.t(2)
        elif action == 6:
            actions_gate.append('CNOT_01')
            circuit.cx(0,1)
        elif action == 7:
            actions_gate.append('MEASURE_0')
            circuit.measure(0, 0) 
        elif action == 8:
            actions_gate.append('MEASURE_1')
            circuit.measure(1, 1) 
        elif action == 9:
            actions_gate.append('MEASURE_2')
            circuit.measure(2, 2) 

    circuit.measure(2, 2)
    print("\nNumber of actions: {}. Actions (gates): {}.\nQuantum Circuit 1:".format(i, actions_gate))
    print(circuit)

    backend1 = Aer.get_backend('qasm_simulator')
    job1 = execute(circuit, backend=backend1, shots=1000)
    result1 = job1.result()
    measurement1 = result1.get_counts(circuit)
    plot_histogram(measurement1)
    plt.show()