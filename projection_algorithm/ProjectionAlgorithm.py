from time import time

import numpy as np
import scipy.optimize as optimize


def solveSystem(h, b, delta=30,s=0,disp=False):
    "function_docstring"


    idx = np.random.randint(low=0, high=h.shape[1], size=int(s))
    idx.sort()
    # print(idx)
    # print(len(idx), " ", s)
    subset = h[:, idx]

    minus_subset = subset * (-1)

    final_subset = np.hstack((minus_subset, subset))
    b_ub = []
    for i in np.nditer(idx):
        b_ub.append(delta-b[i])
    for i in np.nditer(idx):
        b_ub.append(b[i] + delta)

    A_ub = final_subset

    c = np.ones([A_ub.shape[0]])

    res = optimize.linprog(c=c, A_ub=A_ub.transpose(), b_ub=b_ub, options={"disp": disp})

    # print(time()-t1)
    return res.status


def solveFullSystem(h, b, delta=30):
    "function_docstring"
    r = h.shape[0]
    e = 0.1
    s = ((r * np.log(r)) / r) * np.log(r / e)
    # print("Subset size", s)
    # print(h.shape)

    subset = h[:, :]

    minus_subset = subset * (-1)

    final_subset = np.hstack((minus_subset, subset))
    b_ub = []
    for i in range(0, b.shape[0]):
        b_ub.append(delta - b[i])
    for i in range(0, b.shape[0]):
        b_ub.append(b[i] + delta)

    # c = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    A_ub = final_subset

    c = np.zeros([A_ub.shape[0]]) + 1
    # print('------Shapes------')
    # print(c.shape)
    # print(A_ub.shape)
    # print(len(b_ub))
    # print('A_ub:', A_ub.shape,'f:',len(c), 'b_ub:', b_ub)
    res = optimize.linprog(c=c, A_ub=A_ub.transpose(), b_ub=b_ub, options={"disp": True})
    # if res.status == 2:
    #     print('Status:', res.status)
    return res.status


def solveFullSystem(h, b, delta=30):
    "function_docstring"
    r = h.shape[0]
    e = 0.1
    s = ((r * np.log(r)) / r) * np.log(r / e)
    # print("Subset size", s)
    # print(h.shape)
    idx = np.random.randint(low=0, high=h.shape[1] - 1, size=int(s))
    subset = h[:, :]

    minus_subset = subset * (-1)

    final_subset = np.hstack((minus_subset, subset))
    b_ub = []
    for i in range(0, h.shape[1]):
        b_ub.append(delta - b[i])
    for i in range(0, h.shape[1]):
        b_ub.append(b[i] + delta)

    # c = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    A_ub = final_subset

    c = np.zeros([A_ub.shape[0]]) + 1
    # print('------Shapes------')
    # print(c.shape)
    # print(A_ub.shape)
    # print(len(b_ub))
    # print('A_ub:', A_ub.shape,'f:',len(c), 'b_ub:', b_ub)
    res = optimize.linprog(c=c, A_ub=A_ub.transpose(), b_ub=b_ub, options={"disp": False})
    # if res.status == 2:
    #     print('Status:', res.status)

    return res.status

# def solveSystem(filename):
#     "function_docstring"
#     my_data = genfromtxt(filename, delimiter=',')
#
#     r = 4
#     e = 0.1
#     s = ((r * np.log(r)) / r) * np.log(r / e)
#     print(s)
#     a = my_data
#     idx = np.random.randint(a.shape[0], size=int(s))
#     subset = a[idx, :]
#     subset = np.transpose(subset)
#     b = np.random.rand(my_data.shape[1])
#     print(b.shape)
#     print(a.shape)
#     print(subset.shape)
#     c = np.linalg.lstsq(subset, b)
#     print(np.linalg.norm(subset.dot(c[0]) - b))
#
#     return c
