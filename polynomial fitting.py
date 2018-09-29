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
random.shuffle(a)#打乱数组a的顺序  训练集：开发集：测试集=8：1：1
for i in range(0,xnum):
    b.append(func(a[i]))
#创建X,T,W矩阵
m = 7#M阶数，可以更改
n = len(a)
divide = int(n/10*8)#训练集：开发集：测试集=8：1：1
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
X1 = X[0:divide,0:]#训练集
X2 = X[divide:,0:]#开发集
T1 = T[0:divide,0:]#训练集
T2 = T[divide:,0:]#开发集

#第二部分


minlamda = math.exp(-15)
#解析解
def Mode1_1():
    W = (X1.T * X1).I * X1.T * T1  # 不加惩罚项
    return W
def Mode1_2():
    minloss = 10000
    mindex = 0
    Temp = np.mat(np.ones(shape=(m + 1, 1)))
    for i in range(-100,100):
        lamda = math.exp(i)
        I = lamda*np.eye(m + 1)  # 加惩罚项
        W = (X1.T * X1 + I).I * X1.T * T1
        loss = 1/(2*(n-divide)) * ( (X2*W-T2).T*(X2*W-T2)+lamda*W.T*W )
        if(loss<minloss):
            minloss = loss
            mindex = i
            Temp = W
    print("MinLoss=",minloss," Lamda=exp(",mindex,")")
    W = Temp
    return W

#梯度下降
def Mode2_1():
    alpha = 0.08  # 学习率
    iterNum = 1000000  # 迭代次数
    W = np.mat(np.ones(shape=(m + 1, 1)))
    for i in range(0, iterNum):
        # E(W)对矩阵W的一阶导数,结果是一个矩阵
        Differential = 1.0 / divide * (X1.T * X1 * W - X1.T * T1)# 不加惩罚项
        loss = 1 / (2 * divide) * ((X1 * W - T1).T * (X1 * W - T1) + minlamda * W.T * W)
        if(loss<0.015):
            print("迭代次数:", i)
            break
        W = W - alpha * Differential
    return W
def Mode2_2():
    alpha = 0.08  # 学习率
    iterNum = 1000000  # 迭代次数
    W = np.mat(np.ones(shape=(m + 1, 1)))
    for i in range(0, iterNum):
        # E(W)对矩阵W的一阶导数,结果是一个矩阵
        Differential = 1.0 / divide * (X1.T * X1 * W - X1.T * T1 + minlamda * W)# 加惩罚项
        loss = 1 / (2 * divide) * ((X1 * W - T1).T * (X1 * W - T1) + minlamda * W.T * W)
        if (loss < 0.015):
            print("迭代次数:",i)
            break
        W = W - alpha * Differential
    print(" Lamda=", minlamda)
    return W

#共轭梯度
def Mode3_1():
    A = X1.T * X1
    B = X1.T * T1
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
    I = minlamda*np.eye(m + 1)  # 加惩罚项
    A = X1.T*X1 + I
    B = X1.T * T1
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
    print(" Lamda=", minlamda)
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
    plt.scatter(a[0:divide], b[0:divide], c='c', alpha=0.4)
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


