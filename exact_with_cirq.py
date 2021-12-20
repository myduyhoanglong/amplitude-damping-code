from utils import *
import time
import multiprocessing
import sys

np.set_printoptions(threshold=sys.maxsize)


# import sys
# np.set_printoptions(threshold=sys.maxsize)


# def main():
#     p = 0.001
#     count = 0
#     n = 20
#     start = time.time()
#     for i in range(n):
#         perfect_state = initialize()
#         noisy_state = simulate(add_qubits(perfect_state, 6), p_noise=p)
#         noisy_state = rest(noisy_state, p_noise=p)
#         noisy_state = simulate(add_qubits(noisy_state, 6), p_noise=p)
#         try:
#             ec_state = ideal_decoder(add_qubits(noisy_state, 2))
#             infide = infidelity(ec_state, perfect_state)
#         except:
#             print(noisy_state)
#             infide = 1000
#         if infide > p:
#             count += 1
#             print("Infidelity: %f" % infide)
#     end = time.time()
#     print("Count: %d/%d, Time: %f" % (count, n, end - start))

def main():
    power = 4
    p = multiprocessing.Value('d', 10 ** (-power))
    count1 = multiprocessing.Value('i', 0)
    count2 = multiprocessing.Value('i', 0)
    N = 100
    n_per_pool = 1
    n_pools = int(N / n_per_pool)
    jobs = []

    print('Running...')

    start = time.time()
    for po in range(n_pools):
        start_p = time.time()
        for i in range(n_per_pool):
            process = multiprocessing.Process(target=one_run, args=(p.value, count1, count2, power))
            jobs.append(process)
            process.start()

        for proc in jobs:
            proc.join()
        print("Count: %d/%d, Time: %f" % (count1.value, (po + 1) * n_per_pool, time.time() - start_p))
    print("Total count: %d/%d, Time: %f" % (count1.value, N, time.time() - start))
    with open('./count.txt', 'a') as writer:
        writer.write('Error_rate: %f, Count1: %d, Count2: %d, Total: %d, Time: %f\n' % (
            p.value, count1.value, count2.value, N, time.time() - start))


def one_run(p, count1, count2, power):
    np.random.seed()
    q = []
    for i in range(10):
        q.append(cirq.NamedQubit(str(i)))

    simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)

    try:
        perfect_state, r, theta, phi = initialize_randomly()
        noisy_state1, log1 = simulate(simulator, q, add_qubits(perfect_state, 6), p_noise=p)
        noisy_state2 = rest(simulator, q, noisy_state1, p_noise=p)
        noisy_state3, log3 = simulate(simulator, q, add_qubits(noisy_state2, 6), p_noise=p)

        ec_state, logid3 = ideal_decoder(simulator, q, add_qubits(noisy_state3, 2))
        in_state, logid1 = ideal_decoder(simulator, q, add_qubits(noisy_state1, 2))

        perfect_bloch = get_bloch_vector(perfect_state)
        in_bloch = get_bloch_vector(in_state)
        ec_bloch = get_bloch_vector(ec_state)

        infide_perfect = infidelity_no_abs(ec_state, perfect_state)
        infide_perfect_bloch = infidelity_bloch(ec_bloch, perfect_bloch)
        infide = infidelity_no_abs(ec_state, in_state)
        infide_bloch = infidelity_bloch(ec_bloch, in_bloch)
        infide_middle = infidelity_no_abs(in_state, perfect_state)
        infide_middle_bloch = infidelity_bloch(in_bloch, perfect_bloch)
        unencoded_infide = unencoded_infidelity_mix(r, theta, phi, p)

        print((r, theta, phi))
        print(perfect_bloch, in_bloch, ec_bloch)
        print(infide_perfect, infide_perfect_bloch)
        print(infide, infide_bloch)
        print(infide_middle, infide_middle_bloch)
        print(unencoded_infide)
        print(log1, log3, logid1, logid3)
    except Exception as e:
        # print(noisy_state)
        print(e)
        infide = 1000

    filename = './data_mix_' + str(int(power * 100)) + '.txt'
    # filename = './test1L.txt'
    with open(filename, 'a') as writer:
        writer.write(
            make_str(p, perfect_bloch, in_bloch, ec_bloch, infide_perfect, infide_perfect_bloch, infide, infide_bloch,
                     infide_middle, infide_middle_bloch, unencoded_infide, log1, log3, logid1, logid3))
        if 1 >= infide_bloch > unencoded_infide:
            with count1.get_lock():
                count1.value += 1
        if 1 >= infide_perfect_bloch > unencoded_infide:
            with count2.get_lock():
                count2.value += 1


def initialize(theta, phi, ancilla=0):
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    s = np.copy(0.5 * (idt + nx * sigma_x + ny * sigma_y + nz * sigma_z))
    s = add_qubits(s, ancilla)
    return s, theta, phi


def initialize_randomly(ancilla=0):
    # r = np.random.uniform(0, 1)
    r = 1
    theta = np.pi * np.random.uniform(0, 1)
    phi = 2 * np.pi * np.random.uniform(0, 1)
    nx = r * np.sin(theta) * np.cos(phi)
    ny = r * np.sin(theta) * np.sin(phi)
    nz = r * np.cos(theta)
    s = np.copy(0.5 * (idt + nx * sigma_x + ny * sigma_y + nz * sigma_z))
    s = add_qubits(s, ancilla)
    return s, r, theta, phi


def rest(simulator, q, input_state, p_noise=0.):
    circuit = cirq.Circuit([[cirq.I(q[0]), cirq.I(q[1]), cirq.I(q[2]), cirq.I(q[3])]])
    if p_noise > 0:
        circuit = circuit.with_noise(cirq.amplitude_damp(p_noise))
    result = simulator.simulate(circuit, initial_state=input_state)
    return result.final_density_matrix


def format_outcome(meas, stage, decode=False):
    if stage in ['1.1', '1.2']:
        if decode:
            o = ''.join([str(meas[k][0]) for k in ['4', '5']])
        else:
            o = ''.join([str(meas[k][0]) for k in ['8', '9']])
    elif stage == '1.3':
        if decode:
            o = ''.join([str(meas[k][0]) for k in meas.keys()])
        else:
            o = ''.join([str(meas[k][0]) for k in ['4', '5', '6', '7']])
    else:
        o = ''.join([str(meas[k][0]) for k in meas.keys()])
    return o, len(meas.keys())


def format_state(state, stage, num_meas):
    s = trace_out_last_qubits(state, num_meas)
    if stage == '2':
        extra = int(np.log2(s.shape[0])) - 4
        if extra > 0:
            s = trace_out_last_qubits(s, extra)
    if stage in ['1.1', '1.2']:
        s = add_qubits(s, 2)
    elif stage in ['0', '1', '2', '3']:
        pass
    else:
        s = add_qubits(s, 1)
    return s


def trace_out_last_qubits(state, n):
    s = state
    for i in range(n):
        d = s.shape[0]
        s = np.reshape(s, [int(d / 2), 2, int(d / 2), 2])
        s = np.trace(s, axis1=1, axis2=3)
    return s


def add_qubits(state, n):
    s = 1
    for i in range(n):
        s = np.kron(s, np.outer(u[0], u[0]))
    return np.kron(state, s).astype(np.complex128)


def simulate(simulator, q, initial_state, p_noise=0.):
    output_state = initial_state
    outcome = None
    stage = '0'
    num_meas = 0
    log_outcome = []
    while True:
        # print(stage, outcome)
        circuit, stage = get_circuit(q, stage, outcome)
        # print(stage, outcome, num_meas)
        input_state = format_state(output_state, stage, num_meas)
        if circuit is None:
            break
        if p_noise > 0:
            circuit = circuit.with_noise(cirq.amplitude_damp(p_noise))
        result = simulator.simulate(circuit, initial_state=input_state)
        # print(result.measurements)
        outcome, num_meas = format_outcome(result.measurements, stage)
        output_state = result.final_density_matrix
        log_outcome.append((stage, outcome))
        # print("==============")
    output_state = input_state

    return output_state, log_outcome


def ideal_decoder(simulator, q, initial_state):
    # print("decoder...")
    output_state = initial_state
    outcome = None
    stage = '0'
    num_meas = 0
    log_outcome = []
    while True:
        # print(stage, outcome)
        circuit, stage = get_circuit_ideal_decoder(q, stage, outcome)
        input_state = format_state(output_state, stage, num_meas)
        # print(stage, outcome)
        if circuit is None:
            break
        result = simulator.simulate(circuit, initial_state=input_state)
        # print(result.measurements)
        outcome, num_meas = format_outcome(result.measurements, stage, decode=True)
        log_outcome.append((stage, outcome))
        output_state = result.final_density_matrix
        # print("==============")
    output_state = input_state

    return output_state, log_outcome


if __name__ == '__main__':
    main()
