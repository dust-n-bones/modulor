import numpy as np
from numpy import random


def sparsityMask(input,percentance=0,upper_bounds=1):

    out=np.copy(input)
    d_1,d_2= input.shape
    X_u = np.zeros(input.shape)
    size = int((d_1 * d_2) * (percentance/100))
    print(size)
    for i in range(size):
        x = random.randint(0, input.shape[0])
        y = random.randint(0, input.shape[1])
        value = out[x, y]
        X_u[x, y] = upper_bounds
        out[x, y] = 0
    return out,X_u




def L_ucalc(input,delta=1):
    X_u = np.array(input)
    X_u[X_u > 0] += delta
    X_u[X_u < 0] = 0
    return X_u

