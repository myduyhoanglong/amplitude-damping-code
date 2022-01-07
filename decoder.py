from utils import *
import sys

# np.set_printoptions(threshold=sys.maxsize)


# gate operations: cirq simulate function.
# Measure_last_qubit: Input: valid \rho, Output: \rho_i for each branch, p_i for each branch, trace out the last qubit.
# Array to keep track all branches.

q = []
for i in range(10):
    q.append(cirq.NamedQubit(str(i)))

simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)


def main():
    theta = np.random.uniform()*np.pi
    phi = np.random.uniform()*2*np.pi
    print(theta, phi)
    init = initialize(theta, phi)
    noisy = init
    for i in range(10):
        circuit = cirq.Circuit([[cirq.I(q[0]), cirq.I(q[1]), cirq.I(q[2]), cirq.I(q[3])]])
        circuit = circuit.with_noise(cirq.amplitude_damp(10 ** (-3)))
        noisy = simulator.simulate(circuit, initial_state=noisy).final_density_matrix
    fin = decoder(noisy)
    print(get_bloch_vector(init), get_bloch_vector(fin))
    print(infidelity(fin, init), infidelity_bloch(get_bloch_vector(init), get_bloch_vector(fin)))


def initialize(theta, phi):
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    init_state = np.copy(0.5 * (idt + nx * sigma_x + ny * sigma_y + nz * sigma_z))
    return init_state.astype(np.complex128)


def trace_out_last_qubits(state, n):
    s = state
    for i in range(n):
        d = s.shape[0]
        s = np.reshape(s, [int(d / 2), 2, int(d / 2), 2])
        s = np.trace(s, axis1=1, axis2=3)
    return s


# measure the last qubit in Z basis
def measure_last_qubit(s):
    # print("measure...")
    ndims = np.shape(s)[0]
    pz = np.kron(np.identity(ndims // 2), sz)
    proj0 = (np.identity(ndims) + pz) / 2
    proj1 = (np.identity(ndims) - pz) / 2
    s0 = trace_out_last_qubits(np.matmul(proj0, np.matmul(s, proj0)), 1)
    s1 = trace_out_last_qubits(np.matmul(proj1, np.matmul(s, proj1)), 1)
    p0 = np.trace(s0).real
    p1 = np.trace(s1).real
    if p0 == 0:
        s0 = None
    else:
        s0 = s0 / p0

    if p1 == 0:
        s1 = None
    else:
        s1 = s1 / p1
    # print(p0, p1)
    return [(p0, s0, '0'), (p1, s1, '1')]


def measure_last_two_qubits(s):
    pool = measure_last_qubit(s)
    # print(pool)
    new_pool = []
    for (p, s, o) in pool:
        if p == 0 or s is None:
            continue
        cpool = measure_last_qubit(s)
        for (p1, s1, o1) in cpool:
            if p1 == 0 or s1 is None:
                continue
            new_pool.append((p * p1, s1, o1 + o))
    return new_pool


def add_qubits(state, n):
    s = 1
    for i in range(n):
        s = np.kron(s, np.outer(u[0], u[0]))
    return np.kron(state, s).astype(np.complex128)


def decoder(init_state):
    # stage 0
    # print("Stage 0")
    s0 = simulator.simulate(get_circuit(q, '0'), initial_state=add_qubits(init_state, 2)).final_density_matrix
    pool = measure_last_two_qubits(s0)

    # stage 1
    new_pool = []
    for (p, s, o) in pool:
        if p == 0 or s is None:
            continue
        if o in '00':
            out = simulator.simulate(get_circuit(q, '1', o), initial_state=add_qubits(s, 1)).final_density_matrix
            cpool = measure_last_qubit(out)
            for (p1, s1, o1) in cpool:
                if p1 == 0 or s1 is None:
                    continue
                new_pool.append((p * p1, s1, o + o1))
        if o == '10':
            out = simulator.simulate(get_circuit(q, '1', '10'), initial_state=add_qubits(s, 2)).final_density_matrix
            cpool = measure_last_two_qubits(out)
            for (p1, s1, o1) in cpool:
                if p1 == 0 or s1 is None:
                    continue
                new_pool.append((p * p1, s1, o + o1))
        if o == '01':
            out = simulator.simulate(get_circuit(q, '1', '01'), initial_state=add_qubits(s, 2)).final_density_matrix
            cpool = measure_last_two_qubits(out)
            for (p1, s1, o1) in cpool:
                if p1 == 0 or s1 is None:
                    continue
                new_pool.append((p * p1, s1, o + o1))
        if o == '11':
            out = simulator.simulate(get_circuit(q, '1', '11'), initial_state=add_qubits(s, 1)).final_density_matrix
            cpool = measure_last_qubit(out)
            for (p1, s1, o1) in cpool:
                if p1 == 0 or s1 is None:
                    continue
                new_pool.append((p * p1, s1, o + o1))
    pool = new_pool

    # stage 2
    # print("Stage 2")
    new_pool = []
    for (p, s, o) in pool:
        if p == 0 or s is None:
            continue
        if o in ['1001', '1010', '1000', '1011', '0101', '0110', '0100', '0111']:
            out = simulator.simulate(get_circuit(q, '2', o), initial_state=add_qubits(s, 1)).final_density_matrix
            cpool = measure_last_qubit(out)
            for (p1, s1, o1) in cpool:
                if p1 == 0 or s1 is None:
                    continue
                new_pool.append((p * p1, s1, o + o1))
        else:
            new_pool.append((p, s, o))
    pool = new_pool

    # stage 3
    # print("Stage 3")
    new_pool = []
    for (p, s, o) in pool:
        if p == 0 or s is None:
            continue
        if o in ['001', '111', '10011', '10101', '10001', '10111', '01011', '01101', '01001', '01111']:
            out = simulator.simulate(get_circuit(q, '3', o), initial_state=s).final_density_matrix
            new_pool.append((p, out, o))
        else:
            new_pool.append((p, s, o))
    pool = new_pool

    # finalize
    final_state = 0
    total_p = 0
    # print("Decoder...")
    for (p, s, o) in pool:
        # print(p, o, infidelity_pure(s, initialize(np.pi/2, 0)))
        total_p += p
        final_state = final_state + p * s
    # print(total_p)

    return final_state


def get_circuit(q, stage, outcome=''):
    circuit = None
    if stage == '0':
        circuit = [[cirq.CNOT(q[0], q[4]), cirq.CNOT(q[3], q[5])],
                   [cirq.CNOT(q[1], q[4]), cirq.CNOT(q[2], q[5])]]

    if stage == '1':
        if outcome == '00':
            circuit = [[cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])], [cirq.CNOT(q[4], q[2])],
                       [cirq.CNOT(q[4], q[3])], [cirq.H(q[4])]]
        if outcome == '10':
            circuit = [[cirq.CNOT(q[0], q[4]), cirq.CNOT(q[1], q[5]), cirq.I(q[2]), cirq.I(q[3])]]
        if outcome == '01':
            circuit = [[cirq.CNOT(q[2], q[4]), cirq.CNOT(q[3], q[5]), cirq.I(q[0]), cirq.I(q[1])]]
        if outcome == '11':
            circuit = [[cirq.X(q[0]), cirq.X(q[2]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])],
                       [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4])]]

    if stage == '2':
        if outcome == '1001':
            circuit = [[cirq.X(q[0]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4])]]
        if outcome == '1010':
            circuit = [[cirq.X(q[1]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4])]]
        if outcome == '0101':
            circuit = [[cirq.X(q[2]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4])]]
        if outcome == '0110':
            circuit = [[cirq.X(q[3]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4])]]
        if outcome in ['1000', '1011', '0100', '0111']:
            circuit = [[cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), ]]

    if stage == '3':
        if outcome in ['001', '111', '10011', '10101', '10001', '10111']:
            circuit = [[cirq.Z(q[0]), cirq.I(q[1]), cirq.I(q[2]), cirq.I(q[3])]]
        if outcome in ['01011', '01101', '01001', '01111']:
            circuit = [[cirq.Z(q[2]), cirq.I(q[0]), cirq.I(q[1]), cirq.I(q[3])]]

    if circuit is not None:
        circuit = cirq.Circuit(circuit)

    return circuit


if __name__ == "__main__":
    main()
