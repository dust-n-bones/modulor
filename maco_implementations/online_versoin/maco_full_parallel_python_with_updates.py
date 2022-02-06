from time import time
import sys
import numpy as np
from scipy import linalg as lg, random
from multiprocessing import Process
import multiprocessing
from plots.maco_coverage import coverage

import SharedArray as sa
import warnings



def copy_to_shared_Array(np_array,name="name", dtype=float):
    # array= Array('d' ,np_array.shape)
    array = sa.create("shm://"+name,np_array.shape,dtype=dtype)
    entries = np.where(np_array>0)
    size = entries[0].shape[0]
    for i in range(0,size):
            array[entries[0][i],entries[1][i]]=np_array[entries[0][i],entries[1][i]]
    return array


def copy_to_shared_Array_1_dim(np_array,name="name", dtype=float):
    # array= Array('d' ,np_array.shape)
    array = sa.create("shm://"+name,np_array.shape,dtype=dtype)
    entries = np.where(np_array>0)
    for i in range(0,np_array.shape[0]):
            array[i]=np_array[i]
    return array


# def shmem_as_ndarray( np_array,name="name" ):
#
#     """ view processing.Array or processing.Value as ndarray """
#     array_or_value = copy_to_shared_Array(np_array,name)
#     obj = array_or_value._obj
#     buf = obj._wrapper.getView()
#     try:
#         t = _ctypes_to_numpy[type(obj)]
#         return numpy.frombuffer(buf, dtype=t, count=1)
#     except KeyError:
#         t = _ctypes_to_numpy[obj._type_]
#     return numpy.frombuffer(buf, dtype=t)


def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()


def calcE(X):
    return np.where(X > 0)

# Calculate Lipschitz constant for Row update (Eq 6)
def calcRowW(m, R, E_0,E_1, E_unique, r, i):
    W = R[r, E_1[np.where(E_0 == i)]]
    W1 = R[r, E_unique[:, 1][np.where(E_unique[:, 0] == i)]]
    return m + np.sum(np.square(W)) + np.sum(np.square(W1))

# Calculate Lipschitz constant for Column update (Eq 10)
def calcColW(m, L, E_0,E_1, E_unique, r, j):
    W = L[E_0[np.where(E_1 == j)], r]
    W1 = L[E_unique[:, 0][np.where(E_unique[:, 1] == j)], r]
    return m + np.sum(np.square(W)) + np.sum(np.square(W1))



def computeRowDelta(L, R, X,X_u,X_l, m, i, r, E_0,E_1,E_u,E_l,E_unique):
    delta = (m * L[i, r]) + np.sum((np.dot(
        L[i, :].dot(R[:, E_1[np.where(E_0 == i)]]) - X[i, E_1[np.where(E_0 == i)]],
        R[r, E_1[np.where(E_0 == i)]])))

    W= calcRowW(m, R, E_0,E_1, E_unique, r, i)

    l_sum = 0
    u_sum = 0
    for j in E_u[:, 1][np.where(E_u[:, 0] == i)]:
        temp = L[i, :].dot(R[:, j])
        if temp > X_u[i, j]:
            l_sum += (temp - X_u[i, j]) * R[r, j]

    for j in E_l[:, 1][np.where(E_l[:, 0] == i)]:
        temp = L[i, :].dot(R[:, j])
        if temp < X_l[i, j]:
            u_sum += (temp - X_l[i, j]) * R[r, j]

    delta += l_sum + u_sum
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            L[i, r] += -delta / W
        except Warning as e:
            L[i, r] += np.random.sample()

def computeColDelta(L, R, X, X_u ,X_l , m, j, r, E_0, E_1, E_u ,E_l, E_unique):
    delta = (m * R[r, j]) + np.sum((np.dot(
        L[E_0[np.where(E_1 == j)], :].dot(R[:, j]) - X[E_0[np.where(E_1 == j)], j],
        L[E_0[np.where(E_1 == j)], r])))

    W = calcColW(m, L, E_0,E_1, E_unique, r, j)

    l_sum = 0
    u_sum = 0
    for i in E_u[:, 0][np.where(E_u[:, 1] == j)]:

        temp = L[i, :].dot(R[:, j])
        if temp > X_u[i, j]:
            l_sum += (temp - X_u[i, j]) * L[i, r]

    for i in E_l[:, 0][np.where(E_l[:, 1] == j)]:
        temp = L[i, :].dot(R[:, j])
        if temp < X_l[i, j]:
            u_sum += (temp - X_l[i, j]) * L[i, r]

    delta += l_sum + u_sum
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            R[r, j] += -delta / W
        except Warning as e:
            R[r,j] += np.random.sample()


def maco(x_train, X_u, X_l, RMSEs, dimension=15, m=0.0001, epoch_n=100, color='b'):
    t0 = time()

    ##### variables for plots
    l_up_run = 0
    r_up_run = 0
    RMSE = []
    dimensions = x_train.shape
    current = sys.maxsize

    # Init E, L, R and upper/low bounds
    E = calcE(x_train)
    E_l = calcE(X_l)
    E_u = calcE(X_u)

    E_l = np.column_stack((E_l[0], E_l[1]))
    E_u = np.column_stack((E_u[0], E_u[1]))
    E_lu = np.concatenate([E_l, E_u], axis=0)
    E_unique = np.unique(E_lu, axis=0)
   # L, R = pickle.load(open("L-R.p",'rb'))
    L = np.random.rand(dimensions[0], dimension)
    R = np.random.rand(dimension, dimensions[1])
    L[L == 0] = 0.01
    R[R == 0] = 0.01
    for i in sa.list():
        sa.delete("shm://" + str(i[0].decode('UTF-8')))


    E_0 = copy_to_shared_Array_1_dim(E[0],"E_0",dtype=int)
    E_1 = copy_to_shared_Array_1_dim(E[1],"E_1",dtype=int)
    E_l = copy_to_shared_Array(E_l,"E_l",dtype=int)
    E_u= copy_to_shared_Array(E_u,"E_u",dtype=int)
    E_unique = copy_to_shared_Array(E_unique,"E_unique",dtype=int)
    # L and R are initiated with random values
    L= copy_to_shared_Array(L,"L")
    R= copy_to_shared_Array(R, "R")
    #

    iter = 0
    # Alg line 2
    try:

        while (l_up_run + r_up_run) / (x_train.shape[0] + x_train.shape[1]) <epoch_n:
            # for iter in range(0, iterations):  # Alg. line 2
            t1 = time()
            iter += 1
            # Alg line 5
            r = random.randint(0, dimension)
            proc = []
            idx = np.random.randint(low=0, high=L.shape[0], size=int(multiprocessing.cpu_count()))
            for i in np.nditer(idx):
                l_up_run += 1
                p = Process(target=computeRowDelta, args=(L, R, x_train, X_u, X_l, m, i, r, E_0, E_1, E_u, E_l, E_unique))
                proc.append(p)
                p.start()

            for pr in proc:
                pr.join()


            proc = []

            r = random.randint(0, dimension)
            idx = np.random.randint(low=0, high=R.shape[1], size=int(multiprocessing.cpu_count()))
            for j in np.nditer(idx):
                r_up_run += 1
                p = Process(target=computeColDelta, args=(L, R, x_train, X_u, X_l, m, j, r, E_0, E_1, E_u, E_l, E_unique))
                proc.append(p)
                p.start()

            for pr in proc:
                pr.join()


            temp_current = lg.norm(x_train - L.dot(R))

            if temp_current > current:
                m = m / 10
            current = temp_current
            print("Iteration time:", time() - t1)

            print("Reconstruction Error: ", current)
            RMSE.append(current)
            print("Iteration:", iter)



            # current = lg.norm(X - L.dot(R))
        # If does not coverage for 4 continuous times, stop the algorithm
            print("Epoch:", str((l_up_run + r_up_run) / (x_train.shape[0] + x_train.shape[1])))
        # Decrease m
        # m = m * 0.9
    except KeyboardInterrupt:
        pass
    coverage(RMSE)
    temp_current = lg.norm(x_train - L.dot(R))

    current = temp_current
    print("Reconstruction Error: ", current)
    print("Matrix Factorization done in %0.3fs." % (time() - t0))
    print("%0.3fs." % (time() - t0),file=open("times.txt","a"))
    RMSEs.append((RMSE))
    return [L, R]

#
# def findAndDeleteOldValues(E,row):
#     np.delete(E,np.where(E[0]==row)


def macoUpdate(X, X_u, X_l,L,R, rmses, dimension=15, m=0.0001,epoch_n=100, color='r'):
    t0 = time()

    ##### variables for plots
    l_up_run = 0
    r_up_run = 0
    dimensions = X.shape
    current = sys.maxsize
    RMSE = []
    # Init E, L, R and upper/low bounds
    E = calcE(X)
    E_l = calcE(X_l)
    E_u = calcE(X_u)

    E_l = np.column_stack((E_l[0], E_l[1]))
    E_u = np.column_stack((E_u[0], E_u[1]))
    E_lu = np.concatenate([E_l, E_u], axis=0)
    E_unique = np.unique(E_lu, axis=0)
   # L, R = pickle.load(open("L-R.p",'rb'))
   #  for i in sa.list():
   #      print(str(i[0]))
   #      sa.delete("shm://"+ str(i[0].decode('UTF-8')))


    if len(sa.list())>5:
        try:
            sa.delete("shm://" + "E_0")
            sa.delete("shm://" + "E_1")
            sa.delete("shm://" + "E_l")
            sa.delete("shm://" + "E_u")
            sa.delete("shm://" + "E_unique")
        except FileNotFoundError:
            pass


    E_0 = copy_to_shared_Array_1_dim(E[0],"E_0",dtype=int)
    E_1 = copy_to_shared_Array_1_dim(E[1],"E_1",dtype=int)
    E_l = copy_to_shared_Array(E_l,"E_l",dtype=int)
    E_u= copy_to_shared_Array(E_u,"E_u",dtype=int)
    E_unique = copy_to_shared_Array(E_unique,"E_unique",dtype=int)



    # L and R are initiated with random values
    # L= sa.attach("shm://L")
    # R= sa.attach("shm://R")

    # X[row,:]= x_line
    # X_u[row,:]= x_u_line
    # X_l[row,:]= x_l_line
    #

    iter = 0
    print("L-R Norm ",lg.norm(L.dot(R)))
    print("X Norm ",lg.norm(X))
    print("Reconstruction Error Before re-train: ",  lg.norm(X - L.dot(R)))

    # Alg line 2
    try:

        while (l_up_run + r_up_run) / (X.shape[0] + X.shape[1]) <epoch_n:
            # for iter in range(0, iterations):  # Alg. line 2
            t1 = time()
            iter += 1
            # Alg line 5
            r = random.randint(0, dimension)
            proc = []
            idx = np.random.randint(low=0, high=L.shape[0], size=int(multiprocessing.cpu_count()))
            for i in np.nditer(idx):
                l_up_run += 1
                p = Process(target=computeRowDelta,args=(L, R, X,X_u,X_l, m, i, r, E_0,E_1,E_u,E_l, E_unique))
                proc.append(p)
                p.start()

            for pr in proc:
                pr.join()


            proc = []

            r = random.randint(0, dimension)
            idx = np.random.randint(low=0, high=R.shape[1], size=int(multiprocessing.cpu_count()))
            for j in np.nditer(idx):
                r_up_run += 1
                p = Process(target=computeColDelta,args=(L, R, X,X_u,X_l, m, j, r, E_0,E_1, E_u, E_l, E_unique))
                proc.append(p)
                p.start()

            for pr in proc:
                pr.join()


            temp_current = lg.norm(X - L.dot(R))

            if temp_current > current:
                m = m / 10
            current = temp_current
            print(current,m)
            # print("Iteration time:", time() - t1)

            RMSE.append(current)


            # current = lg.norm(X - L.dot(R))
        # If does not coverage for 4 continuous times, stop the algorithm
        #     print("Epoch:", str((l_up_run + r_up_run) / (X.shape[0] + X.shape[1])))
        # Decrease m
        # m = m * 0.9
    except KeyboardInterrupt:
        pass
    coverage(RMSE)
    temp_current = lg.norm(X - L.dot(R))

    current = temp_current
    RMSE.append(current)
    print("Reconstruction Error After Retrain: ", current)
    print("Matrix Factorization done in %0.3fs." % (time() - t0))
    print("%0.3fs." % (time() - t0),file=open("times.txt","a"))
    rmses.append(RMSE)
    return [L, R]


# X = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 0.0, 3.0]])
# X_l=np.zeros(X.shape)
# X_u = np.array([[1.0, 0.1, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 3.0]])
# L,R=maco(X,X_u,X_l,10,m=0.000000000001,epoch_n=1000)
#
# res= L.dot(R)
# print(res)