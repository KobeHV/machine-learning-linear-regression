import numpy as np
import matplotlib.pylab as plt
import random
import math


#第一部分

#取观测值x.y,绘制散点图
a = []
b = []
x=0
xnum = 100#数据集x数目，可以更改
def func(x):
    mu=0#均值
    sigma=0.15#方差
    epsilon = random.gauss(mu,sigma) #高斯分布随机数
    return np.sin(2*np.pi*x)+epsilon
for i in range(0,xnum):
    x=x+1.0/xnum
    a.append(x)
    b.append(func(x))


#创建X,T,W矩阵
m = 10#M阶数，可以更改
n = len(a)
Xa = np.zeros(shape=(n,m+1))
for i in range(0,n):
    for j in range(0,m+1):
        Xa[i,j] = a[i]**j
X = np.mat(Xa)
Ta = np.zeros(shape=(n,1))
for i in range(0,n):
    Ta[i] = b[i]
T = np.mat(Ta)
W = W = np.mat(np.ones(shape=(m + 1, 1)))


#第二部分

#解析解
def Mode1_1():
    W = (X.T * X).I * X.T * T  # 不加惩罚项
    return W
def Mode1_2():
    lamda = math.exp(-10)
    I = np.eye(m + 1)  # 加惩罚项
    Ie = lamda*I
    W = (X.T * X + Ie).I * X.T * T
    return W

#梯度下降
def Mode2_1():
    alpha = 0.05  # 学习率
    iterNum = 100000  # 迭代次数
    W = np.mat(np.ones(shape=(m + 1, 1)))
    for i in range(0, iterNum):
        # E(W)对矩阵W的一阶导数,结果是一个矩阵
        Differential = 1.0 / xnum * (X.T * X * W - X.T * T)# 不加惩罚项
        W = W - alpha * Differential
    return W
def Mode2_2():
    alpha = 0.05  # 学习率
    iterNum = 100000  # 迭代次数
    lamda = math.exp(-8)
    W = np.mat(np.ones(shape=(m + 1, 1)))
    for i in range(0, iterNum):
        # E(W)对矩阵W的一阶导数,结果是一个矩阵
        Differential = 1.0 / xnum * (X.T * X * W - X.T * T + lamda * W)# 加惩罚项
        W = W - alpha * Differential
    return W

#共轭梯度
def Mode3_1():
    A = X.T * X
    B = X.T * T
    W = np.mat(np.ones(shape=(m + 1, 1)))
    R = W
    R = B - A * W
    P = R
    beta = 0
    k = 0
    alfa = 0
    while (k != m):
        R1 = R
        alfa = R.T * R * 1.0 / (P.T * A * P)
        W = W + alfa[0, 0] * P
        R = R - alfa[0, 0] * A * P
        beta = R.T * R * 1.0 / (R1.T * R1)
        P = R + beta[0, 0] * P
        k = k + 1
    return W
def Mode3_2():
    lamda = math.exp(-8)
    I = np.eye(m + 1)  # 加惩罚项
    Ie = lamda * I
    A = X.T*X + Ie
    B = X.T * T
    W = np.mat(np.ones(shape=(m + 1, 1)))
    R = W
    R = B - A * W
    P = R
    beta = 0
    k = 0
    alfa = 0
    while (k != m):
        R1 = R
        alfa = R.T * R * 1.0 / (P.T * A * P)
        W = W + alfa[0, 0] * P
        R = R - alfa[0, 0] * A * P
        beta = R.T * R * 1.0 / (R1.T * R1)
        P = R + beta[0, 0] * P
        k = k + 1
    return W

#第三部分

#拟合多项式函数
def match(x):
    sum = W[0,0]
    for i in range(1,m+1):
        sum += W[i,0]*(x**i)
    return sum

#绘制sin(x)标准曲线和拟合曲线
def draw():
    plt.figure(1, dpi=100)
    plt.figure()
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.scatter(a, b, c='c', alpha=0.4)
    x = np.linspace(0, 1, 100)  # 中间间隔100个元素
    y1 = np.sin(2 * np.pi * x)
    y2 = match(x)
    # 显示LaTex符号,用$\--$表示
    plt.plot(x, y1, color="r", label='sin(2$\pi$x)')
    plt.plot(x, y2, color="g", label='train')
    plt.legend()
    # 去掉右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # 显示所画的图
    plt.show()
    return

#选择模式
choice = 0
while(1):
    choice = input("please input mode:")
    if choice == '1':
        W = Mode1_1()
        draw()
    elif choice == '2':
        W = Mode1_2()
        draw()
    elif choice == '3':
        W = Mode2_1()
        draw()
    elif choice == '4':
        W = Mode2_2()
        draw()
    elif choice == '5':
        W = Mode3_1()
        draw()
    elif choice == '6':
        W = Mode3_2()
        draw()
    elif choice == '-1':
        exit(0)
    else:
        print("Input Error,Please input a number 1-6, -1 is exit")


