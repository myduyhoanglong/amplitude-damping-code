# process tomography

from ec_unit import extended_memory, initialize
from utils import *

# import numpy as np

s0 = np.array([[1., 0], [0, 1]])
s1 = np.array([[0., 1], [1, 0]])
s2 = np.array([[0., -1j], [1j, 0]])
s3 = np.array([[1., 0], [0, -1]])


def main():
    l = 0.5 * np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, -1, 0], [1, 0, 0, -1]]).astype(np.float)
    p_power = [4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0]
    for i in p_power:
        p = 10 ** (-i)
        print(p)
        rho = []
        for (theta, phi) in [(0, 0), (np.pi, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2)]:
            print(theta, phi)
            init = initialize(theta, phi)
            # init_bloch = get_bloch_vector(init)
            fin = extended_memory(init, p_noise=p)
            fin_bloch = get_bloch_vector(fin)
            fin_state = state_from_bloch(fin_bloch)
            rho.append(fin_state)
        s = [rho[0], rho[2] - 1j * rho[3] - 0.5 * (1 - 1j) * (rho[0] + rho[1]),
             rho[2] + 1j * rho[3] - 0.5 * (1 + 1j) * (rho[0] + rho[1]), rho[1]]
        s = np.concatenate((np.concatenate((s[0], s[1]), axis=1), np.concatenate((s[2], s[3]), axis=1)), axis=0)
        chi = np.matmul(l, np.matmul(s, l)).real
        print(chi)


def state_from_bloch(v):
    nx = v[0] * np.sin(v[1]) * np.cos(v[2])
    ny = v[0] * np.sin(v[1]) * np.sin(v[2])
    nz = v[0] * np.cos(v[1])
    s = np.copy(0.5 * (s0 + nx * s1 + ny * s2 + nz * s3))
    return s


if __name__ == "__main__":
    main()
