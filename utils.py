import numpy as np
import cirq
import itertools
from scipy.linalg import sqrtm, eigvalsh

# basis vectors: 0000, 0001, 0010, 0011, ...
u = [np.array([1., 0]), np.array([0, 1])]
sx = np.outer(u[0], u[1]) + np.outer(u[1], u[0])
sz = np.outer(u[0], u[0]) - np.outer(u[1], u[1])
p0 = np.outer(u[0], u[0])
p1 = np.outer(u[1], u[1])
basis = []
for bv in itertools.product([0, 1], repeat=4):
    t = u[bv[0]]
    for i in bv[1:]:
        t = np.kron(t, u[i])
    basis.append(t)
basis = np.array(basis)


def b(s):
    n = int(s, 2)
    return basis[n]


l0 = (b('0000') + b('1111')) / np.sqrt(2)
l1 = (b('0011') + b('1100')) / np.sqrt(2)
sigma_x = np.outer(l0, l1) + np.outer(l1, l0)
sigma_y = -1j * np.outer(l0, l1) + 1j * np.outer(l1, l0)
sigma_z = np.outer(l0, l0) - np.outer(l1, l1)
idt = np.outer(l0, l0) + np.outer(l1, l1)


def infidelity(s1, s2):
    sqrt_s1 = sqrtm(s1)
    eigs = eigvalsh(np.matmul(np.matmul(sqrt_s1, s2), sqrt_s1))
    trace = np.sum(np.sqrt(np.abs(eigs)))
    return 1 - trace ** 2


def infidelity_no_abs(s1, s2):
    sqrt_s1 = sqrtm(s1)
    f = sqrtm(np.matmul(np.matmul(sqrt_s1, s2), sqrt_s1))
    trace = np.trace(f).real
    return 1 - trace ** 2


def infidelity_pure(s1, s2):
    # assume that s2 is pure
    f = np.trace(np.matmul(s1, s2)).real
    return 1 - f


def infidelity_bloch(n1, n2):
    dot = n1[0] * n2[0] * (np.sin(n1[1]) * np.sin(n2[1]) * np.cos(n1[2] - n2[2]) + np.cos(n1[1]) * np.cos(n2[1]))
    f = 0.5 * (1 + dot + np.sqrt((1 - n1[0] * n1[0]) * (1 - n2[0] * n2[0])))
    return 1 - f


def unencoded_infidelity(theta, phi, p):
    return (np.sin(theta / 2) ** 4) * p + (np.sin(theta) ** 2) / 4 * (2 - p - 2 * np.sqrt(1 - p))


def unencoded_infidelity_mix(r, theta, phi, p):
    n = np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])
    n_prime = np.array([np.sqrt(1 - p) * n[0], np.sqrt(1 - p) * n[1], p + (1 - p) * n[2]])
    sq_r_prime = np.sum(np.multiply(n_prime, n_prime))
    f = 0.5 * (1 + np.sum(np.multiply(n, n_prime))) + 0.5 * np.sqrt((1 - r * r) * (1 - sq_r_prime))
    return 1 - f


def log_to_str(log):
    s = '['
    for (st, oc) in log:
        s = s + '(' + st + ',' + oc + ')'
    s = s + ']'
    return s


def list_to_str(l):
    s = '['
    for i in range(len(l) - 1):
        s = s + str(l[i]) + ','
    s = s + str(l[-1]) + ']'
    return s


def make_str(p, perfect_bloch, in_bloch, ec_bloch, infide_perfect, infide_perfect_bloch, infide, infide_bloch,
             infide_middle, infide_middle_bloch, unencoded_infide, log1, log3, logid1, logid3):
    out = "p:%E|perfect:%s|ec1:%s|ec2:%s|if30:%E|if30_b:%E|if31:%E|if31_b:%E|if10:%E|if10_b:%E|if_raw:%E|log1:%s|log3:%s|logid1:%s|logid3:%s\n" % (
        p, list_to_str(perfect_bloch), list_to_str(in_bloch), list_to_str(ec_bloch), infide_perfect,
        infide_perfect_bloch,
        infide, infide_bloch, infide_middle, infide_middle_bloch, unencoded_infide, log_to_str(log1), log_to_str(log3),
        log_to_str(logid1), log_to_str(logid3))
    return out


def project_to_code_space(s):
    pr = np.trace(np.matmul(s, idt))
    s = np.matmul(np.matmul(idt, s), idt)
    return s / pr


def get_bloch_vector(s):
    nx = np.trace(np.matmul(s, sigma_x)).real
    ny = np.trace(np.matmul(s, sigma_y)).real
    nz = np.trace(np.matmul(s, sigma_z)).real
    r = np.sqrt(nx * nx + ny * ny + nz * nz)
    theta = np.arccos(nz / r)
    if nx == 0:
        if ny == 0:
            phi = 0
        else:
            phi = np.pi / 2
    else:
        if nx > 0:
            if ny > 0:
                phi = np.arctan(ny / nx)
            else:
                phi = np.arctan(ny / nx) + 2 * np.pi
        else:
            phi = np.arctan(ny / nx) + np.pi
    return np.array([r, theta, phi])


# stage 1: from the coupling step up to the first two parity measurements.

# stage 1.1: if outcome 10, individual Z-measurements of data qubits 1&2.
# stage 1.1.1: if outcome 01, apply X1 and R1
# stage 1.1.2: if outcome 10, apply X2 and R1
# stage 1.1.3: if outcome 11, R1 only

# stage 1.2: if outcome 01, individual Z-measurements of data qubits 3&4.
# stage 1.2.1: if outcome 01, apply X3 and R1
# stage 1.2.2: if outcome 10, apply X4 and R1
# stage 1.2.3: if outcome 11, R1 only

# stage 1.3: if outcome 00, from M unit up to the measurements of flag qubits.
# stage 3: if outcome 0000, 1111, 0111, 0011, end
# stage 3: if outcome 1100, apply X3X4
# stage 3: if outcome 1110, apply X4
# stage 1.3.(4,5): if outcome 1000, 0100, measure parity of data qubits 1&2
# stage 1.3.(4,5).1: if outcome 0, R1 only
# stage 1.3.(4,5).2: if outcome 1, apply X then R1
# stage 1.3.(6,7): if outcome 0010, 0001, measure parity of data qubits 3&4
# stage 1.3.(6,7).1: if outcome 0, R1 only
# stage 1.3.(6,7).2: if outcome 1, apply X then R1

# stage 2: invalid outcome
# stage 3: end

# Number of adding - tracing qubits
# 1: 0 - 2
# 1.1: 2 - 6
# 1.1.1, 1.1.2, 1.1.3: 1 - 1
# 1.2: 2 - 6
# 1.2.1, 1.2.2, 1.2.3: 1 - 1
# 1.3: 1 - 5
# 1.3.4, 1.3.5, 1.3.6, 1.3.7: 1 - 1
# 1.3.4.1, 1.3.4.2, 1.3.5.1, 1.3.5.2, 1.3.6.1, 1.3.6.2, 1.3.7.1, 1.3.7.2: 1 - 1


# q = []
# for i in range(10):
#     q.append(cirq.NamedQubit(str(i)))
#
# simulator = cirq.DensityMatrixSimulator()


def get_circuit(q, stage, outcome):
    if stage == '-1':
        raise Exception('Invalid stage.')

    new_stage = '-1'
    circuit = None

    if stage == '2' or stage == '3':
        circuit = None
        new_stage = stage

    if stage == '0':
        circuit = [[cirq.CNOT(q[0], q[4]), cirq.CNOT(q[1], q[5]), cirq.CNOT(q[2], q[6]), cirq.CNOT(q[3], q[7])],
                   [cirq.CNOT(q[0], q[8]), cirq.CNOT(q[3], q[9])],
                   [cirq.CNOT(q[1], q[8]), cirq.CNOT(q[2], q[9])],
                   [cirq.measure(q[8]), cirq.measure(q[9])]]
        new_stage = '1'

    if stage == '1':
        if outcome == '10':
            circuit = [[cirq.CNOT(q[0], q[8]), cirq.CNOT(q[1], q[9])],
                       [cirq.measure(q[8]), cirq.measure(q[9])],
                       [cirq.CNOT(q[0], q[4]), cirq.CNOT(q[1], q[5]), cirq.CNOT(q[2], q[6]), cirq.CNOT(q[3], q[7])],
                       [cirq.measure(q[4]), cirq.measure(q[5]), cirq.measure(q[6]), cirq.measure(q[7])]]
            new_stage = '1.1'
        if outcome == '01':
            circuit = [[cirq.CNOT(q[2], q[8]), cirq.CNOT(q[3], q[9])],
                       [cirq.measure(q[8]), cirq.measure(q[9])],
                       [cirq.CNOT(q[0], q[4]), cirq.CNOT(q[1], q[5]), cirq.CNOT(q[2], q[6]), cirq.CNOT(q[3], q[7])],
                       [cirq.measure(q[4]), cirq.measure(q[5]), cirq.measure(q[6]), cirq.measure(q[7])]]
            new_stage = '1.2'
        if outcome == '00':
            circuit = [[cirq.H(q[8])], [cirq.CNOT(q[8], q[0])], [cirq.CNOT(q[8], q[1])], [cirq.CNOT(q[8], q[2])],
                       [cirq.CNOT(q[8], q[3])], [cirq.CNOT(q[8], q[4])], [cirq.CNOT(q[8], q[5])],
                       [cirq.CNOT(q[8], q[6])], [cirq.CNOT(q[8], q[7])], [cirq.H(q[8]), cirq.measure(q[8])],
                       [cirq.CNOT(q[0], q[4]), cirq.CNOT(q[1], q[5]), cirq.CNOT(q[2], q[6]), cirq.CNOT(q[3], q[7])],
                       [cirq.measure(q[4]), cirq.measure(q[5]), cirq.measure(q[6]), cirq.measure(q[7])]]
            new_stage = '1.3'
        if outcome == '11':
            circuit = None
            new_stage = '2'

    if stage == '1.1':
        if outcome == '01':
            circuit = [[cirq.X(q[0]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.1.1'
        if outcome == '10':
            circuit = [[cirq.X(q[1]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.1.2'
        if outcome == '11':
            circuit = [[cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.1.3'
        if outcome == '00':
            circuit = None
            new_stage = '2'

    if stage == '1.2':
        if outcome == '01':
            circuit = [[cirq.X(q[2]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.2.1'
        if outcome == '10':
            circuit = [[cirq.X(q[3]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.2.2'
        if outcome == '11':
            circuit = [[cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.2.3'
        if outcome == '00':
            circuit = None
            new_stage = '2'

    if stage == '1.3':
        if outcome in ['0000', '1111', '0111', '0011']:
            circuit = None
            new_stage = '3'
        elif outcome == '1100':
            circuit = [[cirq.X(q[2]), cirq.X(q[3]), cirq.I(q[0]), cirq.I(q[1])]]
            new_stage = '3'
        elif outcome == '1110':
            circuit = [[cirq.X(q[3]), cirq.I(q[0]), cirq.I(q[1]), cirq.I(q[2])]]
            new_stage = '3'
        elif outcome == '1000':
            circuit = [[cirq.CNOT(q[0], q[4]), cirq.I(q[2]), cirq.I(q[3])], [cirq.CNOT(q[1], q[4])],
                       [cirq.measure(q[4])]]
            new_stage = '1.3.4'
        elif outcome == '0100':
            circuit = [[cirq.CNOT(q[0], q[4]), cirq.I(q[2]), cirq.I(q[3])], [cirq.CNOT(q[1], q[4])],
                       [cirq.measure(q[4])]]
            new_stage = '1.3.5'
        elif outcome == '0010':
            circuit = [[cirq.CNOT(q[2], q[4]), cirq.I(q[0]), cirq.I(q[1])], [cirq.CNOT(q[3], q[4])],
                       [cirq.measure(q[4])]]
            new_stage = '1.3.6'
        elif outcome == '0001':
            circuit = [[cirq.CNOT(q[2], q[4]), cirq.I(q[0]), cirq.I(q[1])], [cirq.CNOT(q[3], q[4])],
                       [cirq.measure(q[4])]]
            new_stage = '1.3.7'
        else:
            circuit = None
            new_stage = '2'

    if stage == '1.3.4':
        if outcome == '0':
            circuit = [[cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.3.4.1'
        if outcome == '1':
            circuit = [[cirq.X(q[0]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.3.4.2'

    if stage == '1.3.5':
        if outcome == '0':
            circuit = [[cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.3.5.1'
        if outcome == '1':
            circuit = [[cirq.X(q[1]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.3.5.2'

    if stage == '1.3.6':
        if outcome == '0':
            circuit = [[cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.3.6.1'
        if outcome == '1':
            circuit = [[cirq.X(q[2]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.3.6.2'

    if stage == '1.3.7':
        if outcome == '0':
            circuit = [[cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.3.7.1'
        if outcome == '1':
            circuit = [[cirq.X(q[3]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.3.7.2'

    if stage in ['1.1.1', '1.1.2', '1.1.3', '1.3.4.1', '1.3.4.2', '1.3.5.1', '1.3.5.2']:
        if outcome == '0':
            circuit = None
            new_stage = '3'
        if outcome == '1':
            circuit = [[cirq.Z(q[0]), cirq.I(q[1]), cirq.I(q[2]), cirq.I(q[3])]]
            new_stage = '3'

    if stage in ['1.2.1', '1.2.2', '1.2.3', '1.3.6.1', '1.3.6.2', '1.3.7.1', '1.3.7.2']:
        if outcome == '0':
            circuit = None
            new_stage = '3'
        if outcome == '1':
            circuit = [[cirq.Z(q[2]), cirq.I(q[0]), cirq.I(q[1]), cirq.I(q[3])]]
            new_stage = '3'

    if circuit is not None:
        circuit = cirq.Circuit(circuit, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    return circuit, new_stage


def get_circuit_ideal_decoder(q, stage, outcome):
    if stage == '-1':
        raise Exception('Invalid stage')

    new_stage = '-1'
    circuit = None

    if stage == '2' or stage == '3':
        circuit = None
        new_stage = stage

    if stage == '0':
        circuit = [[cirq.CNOT(q[0], q[4]), cirq.CNOT(q[3], q[5])],
                   [cirq.CNOT(q[1], q[4]), cirq.CNOT(q[2], q[5])],
                   [cirq.measure(q[4]), cirq.measure(q[5])]]
        new_stage = '1'

    if stage == '1':
        if outcome == '10':
            circuit = [[cirq.CNOT(q[0], q[4]), cirq.CNOT(q[1], q[5]), cirq.I(q[2]), cirq.I(q[3])],
                       [cirq.measure(q[4]), cirq.measure(q[5])]]
            new_stage = '1.1'
        if outcome == '01':
            circuit = [[cirq.CNOT(q[2], q[4]), cirq.CNOT(q[3], q[5]), cirq.I(q[0]), cirq.I(q[1])],
                       [cirq.measure(q[4]), cirq.measure(q[5])]]
            new_stage = '1.2'
        if outcome == '00':
            circuit = [[cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])], [cirq.CNOT(q[4], q[2])],
                       [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.3'
        if outcome == '11':
            circuit = [[cirq.X(q[0]), cirq.X(q[2]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])],
                       [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.3'

    if stage == '1.1':
        if outcome == '01':
            circuit = [[cirq.X(q[0]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.1.1'
        if outcome == '10':
            circuit = [[cirq.X(q[1]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.1.2'
        if outcome == '11':
            circuit = [[cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.1.3'
        if outcome == '00':
            circuit = None
            new_stage = '2'

    if stage == '1.2':
        if outcome == '01':
            circuit = [[cirq.X(q[2]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.2.1'
        if outcome == '10':
            circuit = [[cirq.X(q[3]), cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.2.2'
        if outcome == '11':
            circuit = [[cirq.H(q[4])], [cirq.CNOT(q[4], q[0])], [cirq.CNOT(q[4], q[1])],
                       [cirq.CNOT(q[4], q[2])], [cirq.CNOT(q[4], q[3])], [cirq.H(q[4]), cirq.measure(q[4])]]
            new_stage = '1.2.3'
        if outcome == '00':
            circuit = None
            new_stage = '2'

    if stage in ['1.1.1', '1.1.2', '1.1.3', '1.3']:
        if outcome == '0':
            circuit = None
            new_stage = '3'
        if outcome == '1':
            circuit = [[cirq.Z(q[0]), cirq.I(q[1]), cirq.I(q[2]), cirq.I(q[3])]]
            new_stage = '3'

    if stage in ['1.2.1', '1.2.2', '1.2.3']:
        if outcome == '0':
            circuit = None
            new_stage = '3'
        if outcome == '1':
            circuit = [[cirq.Z(q[2]), cirq.I(q[0]), cirq.I(q[1]), cirq.I(q[3])]]
            new_stage = '3'

    if circuit is not None:
        circuit = cirq.Circuit(circuit)

    return circuit, new_stage
