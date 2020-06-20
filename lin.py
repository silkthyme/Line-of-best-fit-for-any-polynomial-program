import numpy as np
from matplotlib import pyplot as plt


# Goal: Take in matrix m x n and vector of m and spit output vector of thetas n+1 of parameters

def normalize(x):  # normalizes the x matrix by each column
    x_cop = x.copy()
    for i in range(x.shape[1]):
        x_cop[:, i] = (x[:, i] - np.mean(x[:, i])) / np.std(x[:, i])
    return x_cop


def calc_cost(x, y, thetas):  # calculates the cost x : (m x n) y : (m x 1) thetas : (n + 1 x 1)
    x_cop = np.hstack((np.ones((x.shape[0], 1)), x))
    m = x_cop.shape[0]
    n = x_cop.shape[1]
    cost = 0
    for i in range(0, m):
        tran_theta = thetas.T
        tran_x = np.array(x_cop[i, :]).reshape(n, 1)
        cost += (float)(np.matmul(tran_theta, tran_x) - y[i]) ** 2
    return 1 / (2 * m) * cost


def get_thetas(x, y, alpha, thetas,
               iterations):  # takes in same as link, and alpha, thetas of dimension (n+1 x 1) and iterations
    costs = []
    x_cop = np.hstack((np.ones((x.shape[0], 1)), x))
    n = x_cop.shape[1]
    m = x_cop.shape[0]
    copy_theta = thetas.copy()
    for i in range(0, iterations):
        for j in range(0, n):
            tot = 0
            for k in range(0, m):
                tran_theta = thetas.T
                tran_x = np.array(x_cop[k, :]).reshape(n, 1)
                tot += (np.matmul(tran_theta, tran_x) - y[k, 0]) * x_cop[k, j]
            copy_theta[j, 0] = thetas[j, 0] - (alpha * (1 / m) * tot)
        thetas = copy_theta.copy()
        costs.append(calc_cost(x, y, thetas))
    return thetas, costs


def link(x, y, thetas):  # take in x, a matrix of dimension (m x n) and y, a matrix of dimension (m x 1)

    #========Change the stuff here===============#
    alpha = 0.005
    iterations = 1000
    #=============================================#

    return get_thetas(x, y, alpha, thetas, iterations)


def finall():
    print('Hello.')
    degree = input('What kind of graph would you like to do?')

    #========Change the stuff here===================#
    x = np.arange(-1.5, 1.8, 0.1)
    y = [(1.0/7)*(i**8) + 7 - 2*(i**3) for i in x]
    #=================================================#

    x = np.array(x).reshape(len(x), 1)
    block_x = np.tile(x, int(degree))
    for i in range(1, int(degree) + 1):
        block_x[:, i - 1] = block_x[:, i - 1] ** i

    y = np.array(y).reshape(len(y), 1)
    thetas = np.ones((int(degree) + 1, 1))

    thetas, costs = link(block_x, y, thetas)
    block_x = np.hstack((np.ones((block_x.shape[0], 1)), block_x))

    eq = ""
    for i in range(0, int(degree)):
        expon = str(int(degree) - i)
        eq += " " + str(round(thetas[int(degree) - i, 0], 3)) + "$x^{}$.".format(expon) + " + "
    eq += str(round(thetas[0, 0], 3))
    plt.xlabel(eq)
    plt.xlim(min(x) - abs((1. / 10) * min(x)), max(x) + abs((1. / 10) * max(x)))
    plt.ylim(min(y) - abs((1. / 10) * min(y)), max(y) + abs((1. / 10) * max(y)))
    plt.scatter(x, y, label='Actual Points')
    y_test = [np.matmul(thetas.T, block_x[i, :]) for i in range(0, len(x))]
    plt.plot(x, y_test, label='best fit line')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(np.arange(0, len(costs)), costs)
    plt.xlabel('iteratio')
    plt.ylabel('cost')
    plt.show()


finall()
