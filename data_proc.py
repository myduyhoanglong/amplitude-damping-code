import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import re
import ast


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
    print((np.nansum(s[:5]) + np.nansum(s[10:])) / (np.sum(hist[:5]) + np.sum(hist[10:])))
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
                data[i].append(float(v))
    data[0] = data[0][0]

    cnt = 0
    for i in range(n):
        if data[7][i] > data[10][i]:
            cnt += 1
            print(data[6][i], data[7][i], data[10][i])
    print(data[0], cnt, n)
    return data


process_file_2('./dat/data_mix_380.txt')