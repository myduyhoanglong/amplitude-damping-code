# process tomography

from utils import infidelity_bloch
import numpy as np
import ast
import re


# data[0]: p
# data[1]: perfect_bloch
# data[2]: ec1_bloch
# data[3]: ec2_bloch
# data[4]: if30
# data[5]: if30_bloch
# data[6]: if31
# data[7]: if31_bloch
# data[8]: if10
# data[9]: if10_bloch
# data[10]: if_raw
# data[11]: log1
# data[12]: log3
# data[13]: logid1
# data[14]: logid3

def process_file(filename):
    lines = open(filename, 'r').read().splitlines()
    data = [[] for i in range(15)]
    n = 0
    for line in lines:
        n += 1
        its = re.split(', (?!\\d)', line)
        for i in range(len(its)):
            k, v = its[i].split(':')
            if k in ['perfect', 'ec1', 'ec2']:
                data[i].append(ast.literal_eval(v))
            elif k in ['log1', 'log3', 'logid1', 'logid3']:
                data[i].append(v)
            else:
                data[i].append(np.float64(v))
    data[0] = data[0][0]

    final_states = []
    for i in range(n):
        final_bloch_des = sphere_to_descartes(data[3][i])
        final_states.append(final_bloch_des)
    final_state = np.average(np.array(final_states), axis=0)
    init_state = sphere_to_descartes(np.average(np.array(data[1]), axis=0))
    avg_inf = np.average(data[5])

    print(final_state, init_state, infidelity_bloch_descartes(init_state, final_state), avg_inf)
    return data[0], final_state


def infidelity_bloch_descartes(n1, n2):
    dot = np.sum(np.multiply(n1, n2))
    sqr1 = np.sum(np.multiply(n1, n1))
    sqr2 = np.sum(np.multiply(n2, n2))
    f = 0.5 * (1 + dot + np.sqrt((1 - sqr1) * (1 - sqr2)))
    return 1 - f


def sphere_to_descartes(s):
    nx = s[0] * np.sin(s[1]) * np.cos(s[2])
    ny = s[0] * np.sin(s[1]) * np.sin(s[2])
    nz = s[0] * np.cos(s[1])
    return np.array([nx, ny, nz])


def descartes_to_sphere(s):
    nx = s[0]
    ny = s[1]
    nz = s[2]
    r = np.sqrt(nx * nx + ny * ny + nz * nz)
    theta = np.arccos(nz / r)
    if nx == 0:
        if ny == 0:
            phi = 0
        else:
            phi = np.pi / 2
    else:
        if nx > 0:
            if ny >= 0:
                phi = np.arctan(ny / nx)
            else:
                phi = np.arctan(ny / nx) + 2 * np.pi
        else:
            phi = np.arctan(ny / nx) + np.pi
    return np.array([r, theta, phi])


process_file('./dat/data_plus_360.txt')
