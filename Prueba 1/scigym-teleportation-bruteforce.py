import numpy as np
import scigym
import cirq
from cirq.ops import H, T, CNOT, measure
from cirq.circuits import InsertStrategy

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
    q0, q1, q2 = [cirq.GridQubit(i, 0) for i in range(3)]
    circuit = cirq.Circuit()
    for action in actions_int:
        if action == 0:
            actions_gate.append('H_0')
            circuit.append([H(q0)], strategy=InsertStrategy.NEW)
        elif action == 1:
            actions_gate.append('T_0')
            circuit.append([T(q0)], strategy=InsertStrategy.NEW)
        elif action == 2:
            actions_gate.append('H_1')
            circuit.append([H(q1)], strategy=InsertStrategy.NEW)
        elif action == 3:
            actions_gate.append('T_1')
            circuit.append([T(q1)], strategy=InsertStrategy.NEW)
        elif action == 4:
            actions_gate.append('H_2')
            circuit.append([H(q2)], strategy=InsertStrategy.NEW)
        elif action == 5:
            actions_gate.append('T_2')
            circuit.append([T(q2)], strategy=InsertStrategy.NEW)
        elif action == 6:
            actions_gate.append('CNOT_01')
            circuit.append([CNOT(q0, q1)], strategy=InsertStrategy.NEW)
        elif action == 7:
            actions_gate.append('MEASURE_0')
            circuit.append([measure(q0)], strategy=InsertStrategy.NEW)
        elif action == 8:
            actions_gate.append('MEASURE_1')
            circuit.append([measure(q1)], strategy=InsertStrategy.NEW)
        elif action == 9:
            actions_gate.append('MEASURE_2')
            circuit.append([measure(q2)], strategy=InsertStrategy.NEW)            

    print("\nNumber of actions: {}. Actions (gates): {}.\nQuantum Circuit:".format(i, actions_gate))
    print(circuit)