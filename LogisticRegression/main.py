import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadCSV (filename):
    with open (filename, "r") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
    return np.array(dataset)


def normalize(X):
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)
    rng = max - min
    norm_X = 1 - ((max - X) / rng)
    return norm_X


def logistic_func(beta,X):
    return 1.0/(1.0+np.exp(-np.dot(X,beta.T)))


def log_gradient(beta, X, y):
    first_calc = logistic_func(beta, X) - y.reshape(X.shape[0], -1)
    final_calc = np.dot(first_calc.T, X)
    return final_calc

def cost_func(beta, X, y):
    log_func_v = logistic_func(beta, X)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1 - y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return np.mean(final)

def grad(X,y,beta,lr = .01, converge_change = .001):
    cost = cost_func(beta, X, y)
    change_cost = 1
    num_iter = 1

    while(change_cost > converge_change):
        old_cost = cost
        beta = beta - (lr * log_gradient(beta, X, y))
        cost = cost_func(beta, X, y)
        change_cost = old_cost - cost
        num_iter += 1
    return beta,num_iter

def pred_values(beta,X):
    pred_prob = logistic_func(beta,X)
    pred_value = np.where(pred_prob>0.5,1,0)
    return np.squeeze(pred_value)

def plot_reg(X, y, beta):
    x_0 = X[np.where(y == 0.0)][:,1:3]
    x_1 = X[np.where(y == 1.0)][:,1:3]

    print("x_0 shape: ", x_0.shape)
    print("x_1 shape: ", x_1.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_0[:, 0], np.zeros_like(x_0[:, 0]), c='b', label='y = 0')
    ax.scatter(x_1[:, 0], np.zeros_like(x_1[:, 0]), c='r', label='y = 1')

    x1 = np.arange(0, 1, 0.1)
    x2 = -(beta[0, 0] + beta[0, 1] * x1) / beta[0, 2]
    ax.plot(x1, x2,np.zeros_like(x1), c='Green', label='Regression Line')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('z')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dataset = loadCSV("dataset1.csv")
    X = normalize(dataset[:,:-1])
    X = np.hstack((np.matrix(np.ones(X.shape[0])).T,X))
    y = dataset[:,-1]
    beta = np.matrix(np.zeros(X.shape[1]))
    beta,num_iter = grad(X,y,beta)
    print("Yineleme Sayısı: ",num_iter)
    print("Tahmini Regresyon Sayısı: ",beta)
    y_pred = pred_values(beta,X)
    print("Doğru Tahmin Edilen Degerler: ",np.sum(y == y_pred))
    plot_reg(X,y,beta)
