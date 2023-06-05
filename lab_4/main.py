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
            data1.append(float(row[0]))
    return data1

def plot_interval(y, x, color='b', label1=""):
    if (x == 1):
        plt.vlines(x, y[0], y[1], color, lw=1, label = label1)
    else:
        plt.vlines(x, y[0], y[1], color, lw=1)

def plotIntervals():
    data1 = load_csv('Chanel_1_400nm_2mm.csv')
    data_n = [t for t in range(1, len(data1) + 1)]
    data1 = [[data1[i] - EPS, data1[i] + EPS] for i in range(len(data1))]

    for i in range(len(data1)):
        plt.vlines(data_n[i], data1[i][0], data1[i][1], colors = "b", lw = 1)

    plt.title('Data intervals')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("intervals_1.png")
    plt.figure()

def plotIntervalsWithLine():
    data1 = load_csv('Chanel_1_400nm_2mm.csv')
    data_n = [t for t in range(1, len(data1) + 1)]
    data1 = [[data1[i] - EPS, data1[i] + EPS] for i in range(len(data1))]

    for i in range(len(data1)):
        plt.vlines(data_n[i], data1[i][0], data1[i][1], colors = "b", lw = 1)

    y = []
    for i in range(len(data1)):
        y.append(0.91916 + 6.2333e-06 * data_n[i])
    plt.plot(data_n, y, 'r')
    plt.title('Data intervals')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("intervals_2.png")
    plt.figure()

def plotModyfiedIntervalsWithLine():
    data1 = load_csv('Chanel_1_400nm_2mm.csv')
    w = load_csv('w.csv')
    data_n = [t for t in range(1, len(data1) + 1)]
    data1 = [[data1[i] - EPS, data1[i] + EPS] for i in range(len(data1))]

    for i in range(len(data1)):
        plt.vlines(data_n[i], data1[i][0], data1[i][1], colors = "b", lw = 1)

    y = []
    for i in range(len(data1)):
        y.append(9.1921e-01 + 5.6971e-06 * data_n[i])

    for i in range(len(data1)):
        plt.vlines(data_n[i], (data1[i][0] + data1[i][1]) / 2 - w[i] * EPS, (data1[i][0] + data1[i][1]) / 2 + w[i] * EPS, colors = "y", lw = 1)

    plt.plot(data_n, y, 'r')
    plt.title('Data intervals')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("intervals_3.png")
    plt.figure()

def plotOmega():
    w0 = load_csv('w0.csv')
    w = load_csv('w.csv')
    data_n = [t for t in range(1, len(w) + 1)]

    plt.plot(data_n, w0, 'r')
    plt.plot(data_n, w, 'b')
    plt.title('Data intervals')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("omega.png")
    plt.figure()

def plotRegressionResiduals():
    data1 = load_csv('Chanel_1_400nm_2mm.csv')
    data_n = [t for t in range(1, len(data1) + 1)]

    y0 = []
    y1 = []
    y2 = []
    for i in range(len(data1)):
        y0.append(0.91916 + 6.2333e-06 * data_n[i])
        y1.append(9.1921e-01 + 5.6971e-06 * data_n[i])
        y2.append(0)

    for i in range(len(data1)):
        plt.vlines(data_n[i], data1[i] - y0[i] - EPS, data1[i] - y0[i] + EPS, colors="b", lw=1)
    plt.plot(data_n, y2, 'r')
    plt.title('Data intervals')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("regression_1.png")
    plt.figure()

    for i in range(len(data1)):
        plt.vlines(data_n[i], data1[i] - y1[i] - EPS, data1[i] - y1[i] + EPS, colors="b", lw=1)
    plt.plot(data_n, y2, 'r')
    plt.title('Data intervals')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("regression_2.png")
    plt.figure()

def findMu(data1, interval):
    count = 0
    for i in range(len(data1)):
        if ((data1[i][1]) > (interval[1])) and ((data1[i][0]) < (interval[0])):
            count = count + 1
    return count

def findAllMu(data1, z):
    mu = []
    for i in range(len(z)):
        mu.append(findMu(data1, z[i]))
    return mu

def plotMu():
    data1 = load_csv('Chanel_1_400nm_2mm.csv')
    data_n = [t for t in range(1, len(data1) + 1)]

    y0 = []
    y1 = []
    for i in range(len(data1)):
        y0.append(0.91916 + 6.2333e-06 * data_n[i])
        y1.append(9.1921e-01 + 5.6971e-06 * data_n[i])
    regression0 = []
    regression1 = []
    for i in range(len(data1)):
        regression0.append([data1[i] - y0[i] - EPS, data1[i] - y0[i] + EPS])
        regression1.append([data1[i] - y1[i] - EPS, data1[i] - y1[i] + EPS])
    tmp_z0 = []
    tmp_z1 = []

    for i in range(len(data1)):
        tmp_z0.append(regression0[i][0])
        tmp_z0.append(regression0[i][1])
        tmp_z1.append(regression1[i][0])
        tmp_z1.append(regression1[i][1])

    tmp_z0.sort()
    tmp_z1.sort()
    z0 = []
    z1 = []
    for i in range(len(tmp_z0) - 1):
        z0.append([tmp_z0[i], tmp_z0[i + 1]])
        z1.append([tmp_z1[i], tmp_z1[i + 1]])

    mu0 = findAllMu(regression0, z0)
    mu1 = findAllMu(regression1, z1)
    mV0 = []
    mV1 = []

    for i in range(len(z0)):
        mV0.append((z0[i][0] + z0[i][1]) / 2)
        mV1.append((z1[i][0] + z1[i][1]) / 2)

    plt.plot(mV0, mu0)
    plt.plot(mV1, mu1)
    plt.title('Mu_calculations')
    plt.xlabel('mV')
    plt.ylabel('mu')
    plt.savefig("input_mu.png")
    plt.figure()

def plotInfSet():
    beta0 = [0.91921, 0.91921, 0.9192, 0.91908, 0.91899, 0.91899, 0.91901, 0.91905, 0.91921]
    beta1 = [5.4802e-06, 6.7974e-06, 6.8769e-06, 7.7739e-06, 8.319e-06, 7.9467e-06, 7.45e-06, 6.2938e-06, 5.4802e-06]

    plt.plot(beta0, beta1, 'bo--')

    plt.vlines(0.91899, 5.4802e-06, 8.319e-06, colors="r", lw=1)
    plt.vlines(0.91921, 5.4802e-06, 8.319e-06, colors="r", lw=1)
    plt.hlines(5.4802e-06, 0.91899, 0.91921, colors="r", lw=1)
    plt.hlines(8.319e-06, 0.91899, 0.91921, colors="r", lw=1)

    plt.title('Information set')
    plt.xlabel('beta0')
    plt.ylabel('beta1')
    plt.savefig("inform_set.png")
    plt.figure()

def plotHallway():
    data1 = load_csv('Chanel_1_400nm_2mm.csv')
    data_n = [t for t in range(1, len(data1) + 1)]
    data1 = [[data1[i] - EPS, data1[i] + EPS] for i in range(len(data1))]

    for i in range(len(data1)):
        plt.vlines(data_n[i], data1[i][0], data1[i][1], colors="b", lw=1)

    y0 = []
    y1 = []


    for i in range(len(data1)):
        y0.append(9.1899e-01 + 5.4802e-06 * data_n[i])
        y1.append(9.1921e-01 + 8.3357e-06 * data_n[i])
    plt.plot(data_n, y0, 'r--')
    plt.plot(data_n, y1, 'r--')
    plt.title('Data intervals')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("intervals_4.png")
    plt.figure()

def plotHallwayMod():
    data1 = load_csv('Chanel_1_400nm_2mm.csv')
    data_n = [t for t in range(1, len(data1) + 1)]
    data_n1 = [t for t in range(-50, len(data1) + 51)]
    data1 = [[data1[i] - EPS, data1[i] + EPS] for i in range(len(data1))]

    for i in range(len(data1)):
        plt.vlines(data_n[i], data1[i][0], data1[i][1], colors="b", lw=1)

    y0 = []
    y1 = []

    for i in range(len(data_n1)):
        y0.append(9.1899e-01 + 5.4802e-06 * data_n1[i])
        y1.append(9.1921e-01 + 8.3357e-06 * data_n1[i])
    plt.plot(data_n1, y0, 'r--')
    plt.plot(data_n1, y1, 'r--')
    plt.title('Data intervals')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("intervals_5.png")
    plt.figure()

def plotSegmentHallway():
    data1 = load_csv('Chanel_1_400nm_2mm.csv')
    data_n = [t for t in range(1, len(data1) + 1)]
    data1 = [[data1[i] - EPS, data1[i] + EPS] for i in range(len(data1))]

    for i in range(len(data1)):
        plt.vlines(data_n[i], data1[i][0], data1[i][1], colors="b", lw=1)

    data_n_1 = [t for t in range(1, 31)]
    data_n_2 = [t for t in range(31, 176)]
    data_n_3 = [t for t in range(176, 200)]
    y_1_0 = []
    y_1_1 = []
    y_2_0 = []
    y_2_1 = []
    y_3_0 = []
    y_3_1 = []

    for i in range(len(data_n_1)):
        y_1_0.append(9.1886e-01 + 4.1968e-07 * data_n_1[i])
        y_1_1.append(9.1922e-01 + 2.6814e-05 * data_n_1[i])
    for i in range(len(data_n_2)):
        y_2_0.append(9.1919e-01 + 4.8140e-06 * data_n_2[i])
        y_2_1.append(9.1931e-01 + 5.8735e-06 * data_n_2[i])
    for i in range(len(data_n_3)):
        y_3_0.append(9.1511e-01 - 3.3640e-06 * data_n_3[i])
        y_3_1.append(9.2102e-01 + 2.8061e-05 * data_n_3[i])
    plt.plot(data_n_1, y_1_0, 'r--')
    plt.plot(data_n_1, y_1_1, 'r--')
    plt.plot(data_n_2, y_2_0, 'r--')
    plt.plot(data_n_2, y_2_1, 'r--')
    #plt.plot(data_n_3, y_3_0, 'r--')
    #plt.plot(data_n_3, y_3_1, 'r--')

    plt.title('Data intervals')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("intervals_6.png")
    plt.figure()

if __name__ == '__main__':
    #plotIntervals()
    #plotIntervalsWithLine()
    #plotModyfiedIntervalsWithLine()
    #plotOmega()
    #plotRegressionResiduals()
    #plotMu()
    #plotInfSet()
    #plotHallway()
    #plotHallwayMod()
    plotSegmentHallway()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/