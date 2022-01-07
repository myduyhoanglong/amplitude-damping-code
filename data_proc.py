# process data: read files, collect statistics, plot

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import re
import ast
from utils import unencoded_infidelity


def process_file(filename, old=True):
    lines = open(filename, 'r').read().splitlines()
    theta = []
    phi = []
    if_perfect = []
    if_no_ec = []
    if_unenc = []
    cnt2 = 0
    for line in lines:
        values = line.split(', ')
        data = []
        for value in values:
            k, v = value.split(': ')
            data.append(v)
        theta.append(float(data[1]))
        phi.append(float(data[2]))
        if_perfect.append(np.abs(float(data[3])))
        if_no_ec.append(np.abs(float(data[4])))
        if old:
            if_unenc.append(float(data[5]))
        else:
            if_unenc.append(float(data[7]))
            if np.abs(float(data[3])) > float(data[7]):
                cnt2 += 1
                # print(float(data[3]), float(data[4]), float(data[5]), float(data[6]), data[8], data[9], data[10],
                #       data[11])

    p = float(data[0])
    arr = if_no_ec
    N = len(arr)
    bin = 5000
    avg_arr = []
    for i in range(int(N / bin)):
        avg = np.average(arr[i * bin:(i + 1) * bin])
        avg_arr.append(avg)
    avg_if = np.average(arr)
    std_if = np.std(avg_arr)
    avg_if_unenc = np.average(if_unenc)
    cnt = 0
    for i in range(N):
        if arr[i] > p:
            cnt += 1
    print(p, avg_if, avg_if_unenc, std_if, cnt, cnt2)

    return [p, avg_if, avg_if_unenc, std_if, cnt, N]


def process_data():
    p = []
    avg_if = []
    avg_if_unenc = []
    avg_if_unenc_exact = []
    std_if = []
    cnt = []
    N = []
    # files = ['330', '340', '350', '360', '372', '374', '376', '376_n', '378', '380', '384', '388']
    files = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    for i in files:
        # filename = './dat/data_1L_' + i + '.txt'
        filename = 'data_no_abs_' + i + '.txt'
        if i in ['330', '340', '350', '360', '376_n']:
            old = False
        else:
            old = True
        data = process_file(filename, old)
        p.append(data[0])
        avg_if.append(data[1])
        avg_if_unenc.append(data[2])
        std_if.append(data[3])
        cnt.append(data[4])
        N.append(data[5])

    zip_list = sorted(zip(p, avg_if, avg_if_unenc, std_if, cnt, N), key=lambda x: x[0])
    p, avg_if, avg_if_unenc, std_if, cnt, N = zip(*zip_list)

    C = 6531
    B = 8104460
    # C = 12073
    theory = C * np.power(p, 2) + B * np.power(p, 3)
    pL = np.divide(cnt, N)

    fig, ax = plt.subplots()
    ax.plot(p, pL, marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=3)
    # ax.plot(p, avg_if, marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=3)
    # ax.errorbar(p, avg_if, std_if, marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=3)
    # ax.plot(p, theory, marker='x', markerfacecolor='red', markersize=7, color='green')
    # ax.plot(p, avg_if_unenc, linestyle='dashed', color='black')
    # ax.plot(p, p, linestyle='dashed', color='red')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()


def plot_at_y(arr, val, **kwargs):
    plt.plot(arr, np.zeros_like(arr) + val, 'x', **kwargs)
    plt.show()


def plot_histogram(arr, bins):
    logbins = np.logspace(-12, 0, bins)
    print(logbins)
    hist, edges = np.histogram(arr, bins=logbins)
    print(hist, np.sum(hist))
    means, edges, x = binned_statistic(arr, arr, bins=logbins)
    print(means)
    s = np.multiply(hist, means)
    avg = []
    for i in range(len(hist)):
        avg.append(np.nansum(s[:i + 1]) / np.sum(hist[:i + 1]))
    print(avg)
    # print((np.nansum(s[:5]) + np.nansum(s[10:])) / (np.sum(hist[:5]) + np.sum(hist[10:])))
    print(np.nansum(s[9:]) / np.sum(hist[9:]))
    # plt.hist(arr, bins=logbins)
    # plt.xscale('log')
    # plt.show()


def mix_avg_if(p):
    IF = []
    for i in range(100000):
        r = np.random.uniform(0, 1)
        # r = 1
        theta = np.pi * np.random.uniform(0, 1)
        s = np.sin(theta)
        c = np.cos(theta)
        IF_one = 0.5 * (1 - np.sqrt(1 - p) * r * r * s * s - p * r * c - (1 - p) * r * r * c * c) - 0.5 * np.sqrt(
            (1 - r * r) * (1 - (1 - p) * r * r * s * s - (p + (1 - p) * r * c) ** 2))
        IF.append(IF_one)
    print(np.average(IF))


def process_file_2(filename):
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
                # print(n)
                data[i].append(np.float64(v))
    data[0] = data[0][0]

    cnt = 0
    for i in range(n):
        if data[5][i] > data[10][i]:
            # print(data[5][i])
            cnt += 1
            # print(data[4][i], data[5][i], data[3][i], data[11][i], data[12][i], data[14][i])
    print(data[0], cnt, n)
    arr = data[5]
    # N = len(arr)
    # bin = 10000
    # avg_arr = []
    # for i in range(int(N / bin)):
    #     avg = np.average(arr[i * bin:(i + 1) * bin])
    #     avg_arr.append(avg)
    # std_inf = np.std(avg_arr)
    avg_inf = np.average(arr)
    print(avg_inf)
    avg_raw = np.average(data[10])
    p_L = float(cnt) / n
    # plot_histogram(arr, 13)
    return data[0], avg_inf, avg_raw, p_L


def process_data_2():
    files = [400, 390, 380, 370, 360, 350, 340, 330]
    avg_inf = []
    avg_raw = []
    p_L = []
    p = []
    std = []
    for i in files:
        filename = './dat/data_plus_' + str(i) + '.txt'
        a, b, c, d = process_file_2(filename)
        p.append(a)
        avg_inf.append(b)
        avg_raw.append(c)
        p_L.append(d)
        # std.append(e)
    p = np.array(p)
    print(p)
    print(avg_inf)
    # avg_if_unenc = (1 + p - np.sqrt(1 - p)) / 4
    avg_if_unenc = unencoded_infidelity(np.pi, 0, p)
    print(avg_if_unenc)
    # C = 6531
    B = 8104460
    C = 12073
    theory = C * np.power(p, 2) + B * np.power(p, 3)

    fig, ax = plt.subplots()
    # ax.plot(p, avg_inf, marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=3)
    # ax.errorbar(p, avg_inf, std, marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=3)
    ax.plot(p, p_L, marker='o', markerfacecolor='yellow', markersize=7, color='skyblue', linewidth=3)
    # ax.plot(p, avg_if_unenc, linestyle='dashed', color='black')
    ax.plot(p, p, linestyle='dashed', color='brown')
    # ax.plot(p, avg_raw, linestyle='dashed', color='green')
    # ax.plot(p, theory, marker='x', markerfacecolor='red', markersize=7, color='green')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()


def plot():
    # one_L
    enc_inf_one = [2.2097740405291333e-06, 3.500759752972371e-06, 5.5453348749567866e-06, 8.782774240212454e-06,
                 1.3907793997569584e-05, 2.2018478213037262e-05, 3.4849255782898325e-05, 5.5137212439659855e-05,
                 8.719681495739096e-05, 0.00013781915295607572, 0.00021767398433425456, 0.0003434857777445677,
                 0.000541391053584861]
    sim_inf_one = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7.441947488467997e-06, 4.231125069578198e-05,
                 0.00010598800679574001, 0.00017742198495161404, 0.0002246301677906966, 0.000362118934959542,
                 0.0006650931810750703]
    # plus_L
    enc_inf_plus = [2.184451873166182e-06, 3.4607501888661574e-06, 5.482169539439141e-06, 8.683152281641071e-06,
                    1.3750874168438898e-05, 2.177170287454011e-05, 3.446196001877233e-05, 5.4530941167807434e-05,
                    8.625084596780574e-05, 0.00013634923698302437, 0.00021540188858570897, 0.00033999717857668976,
                    0.0005360803794498548]
    sim_inf_plus = [np.nan, np.nan, np.nan, np.nan, np.nan, 2.9272531732834826e-05, 5.6056500764218286e-05,
                    2.5929723364557864e-05, 9.390357030193981e-05, 0.00015773090768745476, 0.00024369408865990985,
                    0.0002605804734315431, 0.0003419232842121663]
    # zero_L
    enc_inf_zero = [3.063896930166621e-06, 4.853279668926014e-06, 7.686592037980944e-06, 1.2171764902091375e-05,
                 1.9269662283938338e-05, 3.049788594899905e-05, 4.825118034712528e-05, 7.630405883107816e-05,
                 0.00012059710736200824, 0.00019046271556966143, 0.00030052709907346653, 0.000473644322572242,
                 0.0007453873966563052]
    # minus_L
    enc_inf_minus = [2.184451873166182e-06, 3.4607501888661574e-06, 5.482169539439141e-06, 8.683152281641071e-06,
                     1.3750874168438898e-05, 2.177170287454011e-05, 3.446196001877233e-05, 5.4530941167807434e-05,
                     8.625084596780574e-05, 0.00013634923698302437, 0.00021540188858570897, 0.00033999717857668976,
                     0.0005360803794498548]
    # plus-i_L
    enc_inf_plus_i = [3.723422840384849e-06, 5.898212360300192e-06, 9.342007378565498e-06, 1.479403817095104e-05,
                      2.342291800472296e-05, 3.707481508641308e-05, 5.866382473407317e-05, 9.278478073715402e-05,
                      0.00014667276714719346, 0.00023170048579601588, 0.0003657049506222698, 0.0005765826777325289,
                      0.0009078043670953662]
    p = np.power(np.array([10]), -1 * np.array([4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3]))
    B = 8104460
    C = 12073
    theory = C * np.power(p, 2) + B * np.power(p, 3)

    theta = np.pi / 2
    phi = np.pi / 2
    unenc_inf = unencoded_infidelity(theta, phi, p)

    fig, ax = plt.subplots()
    # ax.plot(p, enc_inf_plus_i, marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=3)
    # ax.plot(p, sim_inf_1, marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=3,
    #         linestyle='dotted')
    # ax.plot(p, unenc_inf, linestyle='dashed', color='black')

    ax.plot(p, enc_inf_zero, marker='o', markerfacecolor='blue', markersize=3, color='skyblue', label='0')
    ax.plot(p, enc_inf_one, marker='o', markerfacecolor='blue', markersize=3, color='red', label='1')
    ax.plot(p, enc_inf_plus, marker='o', markerfacecolor='blue', markersize=3, color='brown', label='+')
    ax.plot(p, enc_inf_minus, marker='o', markerfacecolor='blue', markersize=3, color='yellow', label='-')
    ax.plot(p, enc_inf_plus_i, marker='o', markerfacecolor='blue', markersize=3, color='orange', label='+i')
    ax.plot(p, theory, marker='x', markersize=3, color='green', label='counting')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend(loc="upper left")
    plt.show()


def solve():
    from scipy.optimize import fsolve
    def f(x):
        B = 8104460
        C = 12073
        return C * np.power(x, 2) + B * np.power(x, 3) - (1 + x - np.sqrt(1 - x)) / 4

    root = fsolve(f, np.array([1e-4]))
    print(root)


# process_data_2()
plot()
