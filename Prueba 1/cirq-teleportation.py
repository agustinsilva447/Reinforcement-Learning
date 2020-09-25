import cirq
from cirq.ops import H, T, CNOT, measure
from cirq.circuits import InsertStrategy

q0, q1, q2 = [cirq.GridQubit(i, 0) for i in range(3)]
circuit = cirq.Circuit()

circuit.append([H(q0)], strategy=InsertStrategy.NEW)
circuit.append([H(q1)], strategy=InsertStrategy.NEW)
circuit.append([H(q2)], strategy=InsertStrategy.NEW)
circuit.append([T(q0)], strategy=InsertStrategy.NEW)
circuit.append([T(q1)], strategy=InsertStrategy.NEW)
circuit.append([T(q2)], strategy=InsertStrategy.NEW)
circuit.append([CNOT(q0, q1)], strategy=InsertStrategy.NEW)
circuit.append([measure(q0)], strategy=InsertStrategy.NEW)
circuit.append([measure(q1)], strategy=InsertStrategy.NEW)
circuit.append([measure(q2)], strategy=InsertStrategy.NEW)

print(circuit)