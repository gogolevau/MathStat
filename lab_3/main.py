import csv
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import math

EPS = 1e-4

def load_csv(filename):
    data1 = []
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=";")
        for row in spamreader:
            data1.append([float(row[0]), float(row[1])])
    return data1

def plot_interval(y, x, color='b', label1=""):
    if (x == 1):
        plt.vlines(x, y[0], y[1], color, lw=1, label = label1)
    else:
        plt.vlines(x, y[0], y[1], color, lw=1)

def plotData(data1):
    count = 0
    prev_count = 0
    cur_d = data1[0][0]
    #plt.axis([0, 200, 4.7545, 4.7575])
    for t in data1:
        if t[0] == cur_d:
            count = count + 1
        else:
            plt.plot([prev_count, count], [cur_d, cur_d], 'b', linewidth=2)
            count = count + 1
            prev_count = count
            cur_d = t[0]
    plt.plot([prev_count, count], [cur_d, cur_d], 'b', linewidth=2)
    plt.title('Experiment_data(Chanel_1_400nm_2mm.csv)')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("input_data1.png")
    plt.figure()

def plotIntervals(data1):
    data_n = [t for t in range(1, len(data1) + 1)]
    data1 = [[data1[i][0] - EPS, data1[i][0] + EPS] for i in range(len(data1))]

    for i in range(len(data1)):
        if (data1[i][0] < 0.9196246) and (data1[i][1] > 0.919663):
            plt.vlines(data_n[i], data1[i][0], data1[i][1], colors = "r", lw = 1)
        else:
            plt.vlines(data_n[i], data1[i][0], data1[i][1], colors = "b", lw = 1)
    plt.title('Data intervals')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("intervals_2.png")
    plt.figure()

def plotNewIntervals(data1):
    data_n = [t for t in range(1, len(data1) + 1)]
    data1 = [[data1[i][0] - EPS * 7.7490, data1[i][0] + EPS * 7.7490] for i in range(len(data1))]

    for i in range(len(data1)):
        plt.vlines(data_n[i], data1[i][0], data1[i][1], colors = "b", lw = 1)
    plt.hlines(0.919763, 0, 200, colors="r", lw=2)
    plt.title('Data intervals')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("intervals_3.png")
    plt.figure()

def findMin(data1):
    min = data1[0][0]
    for i in range(len(data1)):
        if data1[i][0] < min:
            min = data1[i][0]
    return min

def findMax(data1):
    max = data1[0][0]
    for i in range(len(data1)):
        if data1[i][0] > max:
            max = data1[i][0]
    return max

def findMax(mu):
    max = mu[0]
    for i in range(len(mu)):
        if mu[i] > max:
            max = mu[i]
    return max

def findWid(min, max):
    return max - min

def findRad(min, max):
    return (max - min) / 2

def findMid(min, max):
    return (max + min) / 2

def findMu(data1, interval):
    count = 0
    for i in range(len(data1)):
        if ((data1[i][0] + EPS) > (interval[1])) and ((data1[i][0] - EPS) < (interval[0])):
            count = count + 1
    return count

def findAllMu(data1, z):
    mu = []
    for i in range(len(z)):
        mu.append(findMu(data1, z[i]))
    return mu

def plotMu(data1):
    up_count = 0
    down_count = 0
    tmp_z = []

    for i in range(2 * len(data1)):
        if down_count < len(data1):
            if (data1[down_count][0] - EPS < data1[up_count][0] + EPS):
                tmp_z.append(data1[down_count][0] - EPS)
                down_count = down_count + 1
            else:
                tmp_z.append(data1[up_count][0] + EPS)
                up_count = up_count + 1
        else:
            tmp_z.append(data1[up_count][0] + EPS)
            up_count = up_count + 1

    z = []
    for i in range(len(tmp_z) - 1):
        z.append([tmp_z[i], tmp_z[i + 1]])

    mu = findAllMu(data1, z)
    mV = []
    mx = findMax(mu)

    maxIntervals = []
    for i in range(len(mu)):
        if mu[i] == mx:
            maxIntervals.append([z[i][0], z[i][1]])
    down_max = findDownMax(maxIntervals)
    up_min = findUpMin(maxIntervals)
    down_min = findDownMax(data1)
    up_max = findUpMin(data1)
    print((up_min - down_max) / (up_max - down_min))
    print(maxIntervals[0][0], maxIntervals[len(maxIntervals) - 1][1])
    plt.axis([0.9189, 0.9206, 0, 55])
    for i in range(len(z)):
        mV.append(findMid(z[i][0], z[i][1]))

    plt.plot(mV, mu)
    plt.hlines(findMax(mu), 0.9189, 0.9206, 'r', '--')
    plt.title('Mu_calculations')
    plt.xlabel('mV')
    plt.ylabel('mu')
    plt.savefig("input_mu.png")
    plt.figure()


def findDownMin(data1):
    min = data1[0][0] - EPS
    for i in range(len(data1)):
        if data1[i][0] - EPS < min:
            min = data1[i][0] - EPS
    return min

def findDownMax(data1):
    max = data1[0][0] - EPS
    for i in range(len(data1)):
        if data1[i][0] - EPS > max:
            max = data1[i][0] - EPS
    return max

def findUpMin(data1):
    min = data1[0][0] + EPS
    for i in range(len(data1)):
        if data1[i][0] + EPS < min:
            min = data1[i][0] + EPS
    return min

def findUpMax(data1):
    max = data1[0][0] + EPS
    for i in range(len(data1)):
        if data1[i][0] + EPS > max:
            max = data1[i][0] + EPS
    return max

def Jac(data1):
    up_max = findUpMax(data1)
    up_min = findUpMin(data1)
    down_max = findDownMax(data1)
    down_min = findDownMin(data1)
    return (up_min - down_max) / (up_max - down_min)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data1 = load_csv('Chanel_1_400nm_2mm.csv')
    #plotData(data1)
    plotIntervals(data1)
    plotMu(data1)
    plotNewIntervals(data1)
    print(Jac(data1))