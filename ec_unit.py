from utils import *
from decoder import decoder
import time

q = []
for i in range(10):
    q.append(cirq.NamedQubit(str(i)))

simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)


def main():
    theta = np.pi/2
    phi = np.pi/2
    print(theta, phi)
    init = initialize(theta, phi)
    init_bloch = get_bloch_vector(init)
    p_noise = [4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3]
    infide_arr = []
    for p in p_noise:
        # fin = ec_unit(init, p_noise=10 ** (-p))
        # fin = decoder(fin)
        fin = extended_memory(init, p_noise=10 ** (-p))
        fin_bloch = get_bloch_vector(fin)
        infide = infidelity_bloch(fin_bloch, init_bloch)
        infide_arr.append(infide)
        print(10 ** (-p), infide, fin_bloch)
    print(infide_arr)


def extended_memory(init_state, p_noise=0.):
    s1 = ec_unit(init_state, p_noise)
    s2 = rest(s1, p_noise)
    s3 = ec_unit(s2, p_noise)
    s4 = decoder(s3)
    return s4


def rest(input_state, p_noise=0.):
    circuit = cirq.Circuit([[cirq.I(q[0]), cirq.I(q[1]), cirq.I(q[2]), cirq.I(q[3])]])
    if p_noise > 0:
        circuit = circuit.with_noise(cirq.amplitude_damp(p_noise))
    result = simulator.simulate(circuit, initial_state=input_state)
    return result.final_density_matrix


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


def add_qubits(state, n):
    if n == 0:
        return state
    s = 1
    for i in range(n):
        s = np.kron(s, np.outer(u[0], u[0]))
    return np.kron(state, s).astype(np.complex128)


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


def measure_last_qubits(s, n):
    if n == 1:
        return measure_last_qubit(s)
    pool = measure_last_qubit(s)
    for i in range(n - 1):
        new_pool = []
        for (p1, s1, o1) in pool:
            if p1 == 0 or s1 is None:
                continue
            cpool = measure_last_qubit(s1)
            for (p2, s2, o2) in cpool:
                if p2 == 0 or s2 is None:
                    continue
                new_pool.append((p1 * p2, s2, o2 + o1))
        pool = new_pool
    return pool


def ec_unit(init_state, p_noise=0.):
    pool = [(1., init_state, '')]
    done = False
    while not done:
        done = True
        new_pool = []
        for (p, s, o) in pool:
            stage = get_stage_from_outcome(o)
            # print(o, stage)
            if stage == '4':
                new_pool.append((p, s, o))
            else:
                out = simulate(s, stage, p_noise)
                cpool = measure(out, stage, p_noise)
                done = False
                if cpool is None:
                    new_pool.append((p, out, o + '-'))
                else:
                    for (p1, s1, o1) in cpool:
                        if p1 == 0 or s1 is None:
                            continue
                        new_pool.append((p * p1, s1, o + o1))
        pool = new_pool

    # finalize
    final_state = 0
    total_p = 0
    for (p, s, o) in pool:
        # decode_s = decoder(s)
        # print(p, o, infidelity_pure(decode_s, initialize(np.pi / 2, 0)))
        final_state = final_state + p * s
        # print(p, o)
        total_p += p
    # print(total_p)

    return final_state


def simulate(s, stage, p_noise=0.):
    circuit, num_add = get_circuit(q, stage)
    if circuit is not None and p_noise > 0:
        circuit = circuit.with_noise(cirq.amplitude_damp(p_noise))
    out = simulator.simulate(circuit, initial_state=add_qubits(s, num_add))
    return out.final_density_matrix


def get_stage_from_outcome(o):
    if o == '':
        stage = '1'
    elif o == '00':
        stage = '1.1'
    elif o == '10':
        stage = '1.2'
    elif o == '01':
        stage = '1.3'
    elif o in ['000', '001', '1010', '1001', '1011', '1000', '0110', '0101', '0111', '0100', '11']:
        stage = '1.1.0'
    elif o in ['0001000', '0011000', '0000100', '0010100']:
        stage = '1.1.0.1'
    elif o in ['0000010', '0010010', '0000001', '0010001']:
        stage = '1.1.0.2'
    elif o in ['0001100', '0011100']:
        stage = '1.1.0.3'
    elif o in ['0001110', '0011110']:
        stage = '1.1.0.4'
    elif (o in ['00010000', '00110000', '00001000', '00101000', '00000100', '00100100', '00000010', '00100010']) or (
            o[:4] in ['1011', '0111'] and len(o) == 8):
        stage = '2.0'
    elif (o in ['00010001', '00110001']) or (o[:4] == '1001' and len(o) == 8):
        stage = '2.1'
    elif (o in ['00001001', '00101001']) or (o[:4] == '1010' and len(o) == 8):
        stage = '2.2'
    elif (o in ['00000101', '00100101']) or (o[:4] == '0101' and len(o) == 8):
        stage = '2.3'
    elif (o in ['00000011', '00100011']) or (o[:4] == '0110' and len(o) == 8):
        stage = '2.4'
    elif (o in ['000100001', '001100001', '000010001', '001010001', '000100011', '001100011', '000010011',
                '001010011']) or (
            o[:4] in ['1011', '1001', '1010'] and o[-1] == '1' and len(o) == 9):
        stage = '3.1'
    elif (o in ['000001001', '001001001', '000001011', '001001011', '000000101', '001000101', '000000111',
                '001000111']) or (
            o[:4] in ['0111', '0101', '0110'] and o[-1] == '1' and len(o) == 9):
        stage = '3.2'
    else:
        stage = '4'

    return stage


def measure(s, stage, p_noise=0.):
    if stage in ['1', '1.2', '1.3']:
        pool = measure_last_qubits(s, 2)
    elif stage in ['1.1', '1.1.0.1', '1.1.0.2', '2.0', '2.1', '2.2', '2.3', '2.4']:
        pool = measure_last_qubit(s)
    elif stage == '1.1.0':
        pool = measure_last_qubits(s, 4)
    else:
        pool = None

    # apply one step of AD for measurement?
    if pool is not None:
        new_pool = []
        for (p, s, o) in pool:
            if p == 0 or s is None:
                continue
            ndims = np.shape(s)[0]
            nq = int(np.log2(ndims))
            circuit = []
            for i in range(nq):
                circuit.append(cirq.I(q[i]))
            circuit = cirq.Circuit([circuit], strategy=cirq.InsertStrategy.NEW_THEN_INLINE).with_noise(
                cirq.amplitude_damp(p_noise))
            out = simulator.simulate(circuit, initial_state=s).final_density_matrix
            new_pool.append((p, out, o))
        pool = new_pool
    return pool


def get_circuit(q, stage):
    circuit = None
    n = 0

    # first two parity measurements
    if stage == '1':
        circuit = [[cirq.CNOT(q[0], q[4]), cirq.CNOT(q[1], q[5]), cirq.CNOT(q[2], q[6]), cirq.CNOT(q[3], q[7])],
                   [cirq.CNOT(q[0], q[8]), cirq.CNOT(q[3], q[9])],
                   [cirq.CNOT(q[1], q[8]), cirq.CNOT(q[2], q[9])]]
        n = 6

    # XX..XX measurement
    if stage == '1.1':
        circuit = [[cirq.H(q[8])], [cirq.CNOT(q[8], q[0])], [cirq.CNOT(q[8], q[1])], [cirq.CNOT(q[8], q[2])],
                   [cirq.CNOT(q[8], q[3])], [cirq.CNOT(q[8], q[4])], [cirq.CNOT(q[8], q[5])],
                   [cirq.CNOT(q[8], q[6])], [cirq.CNOT(q[8], q[7])], [cirq.H(q[8])]]
        n = 1

    # decouple, measure all flags
    if stage == '1.1.0':
        circuit = [[cirq.CNOT(q[0], q[4]), cirq.CNOT(q[1], q[5]), cirq.CNOT(q[2], q[6]), cirq.CNOT(q[3], q[7])]]
        n = 0

    # first two flags triggered, measure first parity
    if stage == '1.1.0.1':
        circuit = [[cirq.CNOT(q[0], q[4]), cirq.I(q[1]), cirq.I(q[2]), cirq.I(q[3])], [cirq.CNOT(q[1], q[4])]]
        n = 1

    # last two flags triggered, measure second parity
    if stage == '1.1.0.2':
        circuit = [[cirq.CNOT(q[2], q[4]), cirq.I(q[0]), cirq.I(q[1]), cirq.I(q[3])], [cirq.CNOT(q[3], q[4])]]
        n = 1

    # flag = 1100
    if stage == '1.1.0.3':
        circuit = [[cirq.X(q[2]), cirq.X(q[3]), cirq.I(q[0]), cirq.I(q[1])]]
        n = 0

    # flag == 1110
    if stage == '1.1.0.4':
        circuit = [[cirq.X(q[3]), cirq.I(q[0]), cirq.I(q[1]), cirq.I(q[2])]]
        n = 0

    # Z1Z2 = 10
    if stage == '1.2':
        circuit = [
            [cirq.CNOT(q[0], q[8]), cirq.CNOT(q[1], q[9]), cirq.I(q[2]), cirq.I(q[3]), cirq.I(q[4]), cirq.I(q[5]),
             cirq.I(q[6]), cirq.I(q[7])]]
        n = 2

    # Z1Z2 = 01
    if stage == '1.3':
        circuit = [
            [cirq.CNOT(q[2], q[8]), cirq.CNOT(q[3], q[9]), cirq.I(q[0]), cirq.I(q[1]), cirq.I(q[4]), cirq.I(q[5]),
             cirq.I(q[6]), cirq.I(q[7])]]
        n = 2

    # R1
    if stage == '2.0':
        circuit = [[cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                   [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4])]]
        n = 1

    # X1+R1
    if stage == '2.1':
        circuit = [[cirq.X(q[0]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                   [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4])]]
        n = 1

    # X2+R1
    if stage == '2.2':
        circuit = [[cirq.X(q[1]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                   [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4])]]
        n = 1

    # X3+R1
    if stage == '2.3':
        circuit = [[cirq.X(q[2]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                   [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4])]]
        n = 1

    # X4+R1
    if stage == '2.4':
        circuit = [[cirq.X(q[3]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                   [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4])]]
        n = 1

    # Z1
    if stage == '3.1':
        circuit = [[cirq.Z(q[0]), cirq.I(q[1]), cirq.I(q[2]), cirq.I(q[3])]]
        n = 0

    # Z3
    if stage == '3.2':
        circuit = [[cirq.Z(q[2]), cirq.I(q[0]), cirq.I(q[1]), cirq.I(q[3])]]
        n = 0

    if circuit is not None:
        circuit = cirq.Circuit(circuit, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    return circuit, n


if __name__ == "__main__":
    main()
