import scigym
import numpy as np

import cirq
from cirq.ops import H, T, CNOT, measure
from cirq.circuits import InsertStrategy

j = 0; reward = 0

while (reward == 0) and (j<1000) :
    env = scigym.make('teleportation-v0')
    if isinstance(env.action_space,scigym.Discrete):
        num_actions = env.action_space.n
    else:
        raise ValueError

    i = 0; j += 1
    done = False
    observation = env.reset()
    actions_int = []; actions_gate = []
    available = {}; available['available_actions'] = range(num_actions)

    q0, q1, q2 = [cirq.GridQubit(i, 0) for i in range(3)]
    circuit = cirq.Circuit()

    while done==False:  
        action = np.random.choice(available['available_actions'])
        actions_int.append(action)
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

        i += 1
        (observation, reward, done, available) = env.step(action)
        print("{}, {} --> Reward: {}. Done: {}. Available actions: {}.".format(j, i, reward, done, available['available_actions']), end=" ")
        
        if bool(available['available_actions']):
            print("Action: {}. Gate: {}".format(action, actions_gate[i-1]))
        else:
            print("No more available actions.")

    if reward == 0:
        print("Target not met.")
    elif reward == 1:
        print("Number of actions: {}. Actions (gates): {}.".format(i, actions_gate))
        #print("Actions (numbs): {}.".format(actions_int))
        #print("Congratulations!")

##########