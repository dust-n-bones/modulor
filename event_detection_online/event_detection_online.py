import sys

from maco_implementations.online_versoin.maco_full_parallel_python_with_updates import maco, macoUpdate
from plots.maco_coverage import multiplecoverage

sys.path.extend(["../matrixcompletionandprojection"])
from time import time
from plots.DeltaVariance import plotValidInvalidPercentage, deltaVarianceChart, deltaVarianceImpactOnFlowChart, \
    new_input_plot
from plots.Recall_Precision_F1Measure import recallPlot, precisionPlot, f1ScorePlot
from plots.distance_plots import avgDistanceChart, plotStandardDeviation, DistanceChart
from projection_algorithm.ProjectionAlgorithm import solveSystem
import pickle
from tools.Tools import L_ucalc
import ctypes
import numpy as np
import multiprocessing
import SharedArray as sa

mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
for i in sa.list():
   print(str(i[0]))
   sa.delete("shm://"+ str(i[0].decode('UTF-8')))

# def mkl_set_num_threads(cores):
#     mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
#
#
# np.__config__.show()
# mkl_set_num_threads(multiprocessing.cpu_count())
#
# X = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 0.0, 3.0]])
# t = np.where(X > 0)
#
# for i in np.nditer(t[0]):
#     for j in np.nditer(t[1]):
#         print(i, " ", j)
#
# X1 = np.array([1.0, 0, 1.1])
# t1 = np.where(X1 > 0)

# print(multiprocessing.cpu_count())
# print("Select the number of sensors you want to keep (type 0 to select all available sensors)")
# number_of_sensors = int(input())
# print("Select the number of rows of the synthetic matrices:")
# count_rows = int(input())
# print("Select [-D,+D] of the synthetic matrices:")
# synth_delta = int(input())
# print("Select mean of the Gaussian noise on the event synthetic matrix")
# synth_mean = int(input())
# print("Select dimension K for the matrix completion")
# dimension = int(input())
# print(
#     "Select percentage of validation size. (E.g. if input size rows=1000, percentage=0.1 will lead to 100 normal and 100 event test samples)")
# test_size = float(input())

# x = MatrixCreator.createOnlySynthMatrixKeepZeros(input_matrix="../stored_matrices/matrix.csv", size=count_rows, Delta=synth_delta,sensors_to_keep=number_of_sensors)
# pickle.dump(x,open("x.p","wb"))
# exit(2)

synth_delta=10
dimension=10

x_full = pickle.load(open("x.p", "rb"))
x_full = x_full[:,:4500]

print("Initial Matrix Contains ", x_full.shape[0], "Rows of data. How many you want to keep on the initial dataset?", )
rows_to_keep = int(input())
if rows_to_keep >= x_full.shape[0]:
    print("Rows to keep sould be <= compare to total rows")
    exit(1)

x_train = x_full[:rows_to_keep, :]
x_test = x_full[-(x_full.shape[0]-rows_to_keep):, :]
max = np.amax(x_train)
x_train = x_train/max
x_test = x_test/max
normalized_delta = synth_delta/max
# noisy = noisy / max

# Factorization with MACO
X_u = L_ucalc(x_train, normalized_delta)
X_l = L_ucalc(x_train, -normalized_delta)

RMSE=[]
# Initial training
L, R = maco(x_train, X_u, X_l, m=0.01, dimension=dimension, epoch_n=1,RMSEs=RMSE)

pickle.dump([L, R], open("L-R.p", "wb"))
L, R  = pickle.load(open("L-R.p", "rb"))


daily_samples = 90
n_components = dimension
percentage = 0.5

# Arrays for various statistics
sensor_avg = []
sensor_max = []
sensor_min = []
samples_Standard_deviation = []
samples_Mean_deviation = []

# initialization of statistic arrays
for i in range(0, int(x_train.shape[1] / daily_samples)):
    index = np.arange(start=i * daily_samples, stop=i * daily_samples + daily_samples)
    sensor_max.append(np.amax(x_train[:, index]))
    sensor_min.append(np.amin(x_train[:, index]))
    sensor_avg.append(np.average(x_train[:, index]))
    samples_Standard_deviation.append(np.std(x_train[:, index]))
    samples_Mean_deviation.append(np.mean(x_train[:, index]))
    # print("Sensor Max:", sensor_max[i], " Min:", sensor_min[i], "Average: ",sensor_avg[i],sep=" ")

deltas_valid = []
deltas_invalid = []
deltas_total = []

positive = []
negative = []
normal_flow = []
event_flow = []
plot_valid_percentage = []
plot_invalid_percentage = []

# iteration parameters
step_n = 10
number_of_samples = x_train.shape[0] / step_n

# validation_matrix = MatrixCreator.createVadilationMatrix(x_train, noisy, percentage, test_size)
# negative_start = validation_matrix.shape[0] * 0.5
#
# print("Normal samples", norm(validation_matrix[:int(negative_start), :]))
# print("Event samples", norm(validation_matrix[int(negative_start):, :]))
recall_list = []
precision_list = []
f1_score_list = []
delta_list = []

row_to_replace = 0
positives = []
negatives = []
print(x_test.shape)
for day in range(0, x_test.shape[0]):
    row = x_test[day, :]
    value = solveSystem(R, row, delta=normalized_delta)
    if value == 0:
        positives.append(day)
    if value == 2:
        negatives.append(day)

    x_train[row_to_replace, :] = row

    # Factorization with MACO
    X_u = L_ucalc(x_train, normalized_delta)

    X_l = L_ucalc(x_train, -normalized_delta)

    # Initial training
    L, R = macoUpdate(x_train, X_u, X_l,m=0.01, dimension=dimension,
                      epoch_n=1,rmses=RMSE,color='r')
    row_to_replace += 1
    row_to_replace = row_to_replace % x_train.shape[0]

pickle.dump(RMSE,open("RMSEs.p",'wb'))

# L, R = pickle.load(open("L-R.p","rb"))
exit(1)