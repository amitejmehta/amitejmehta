import numpy as np
import matplotlib.pyplot as plt

def n(a, b):
    sum = 0
    for x in range(len(a)):
        sum += abs(a[x]) **b
    return sum**(1/b)


for i in [0.5, 1 ,2]:
    X,Y = np.meshgrid(np.linspace(-3,3, 200), np.linspace(-3,3, 200))
    stack = np.dstack((X,Y))
    final = np.zeros((200,200))
    for a in range(200):
        for b in range(200):
            final[a][b] = n(stack[a][b], i)

    plt.figure()
    plt.contourf(X,Y,final)
    plt.colorbar()
    plt.show()


