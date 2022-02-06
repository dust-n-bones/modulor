import sys

import scipy as sp

from maco_implementations.online_versoin.maco_full_parallel_python_with_updates import maco, macoUpdate, copy_to_shared_Array
from plots.maco_coverage import multiplecoverage, coverage
from tools import MatrixCreator

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




print(multiprocessing.cpu_count())
print("Select the number of sensors you want to keep (type 0 to select all available sensors)")
number_of_sensors = int(input())
print("Select the number of rows of the synthetic matrices:")
count_rows = int(input())
print("Select [-D,+D] of the synthetic matrices:")
synth_delta = int(input())
print("Select mean of the Gaussian noise on the event synthetic matrix")
synth_mean = int(input())
print("Select dimension K for the matrix completion")
dimension = int(input())
print(
    "Select percentage of validation size. (E.g. if input size rows=1000, percentage=0.1 will lead to 100 normal and 100 event test samples)")
test_size = float(input())

synth, noisy = MatrixCreator.createIncrementalSynthMatrixKeepZeros(input_matrix="../stored_matrices/matrix.csv", size=count_rows, Delta=synth_delta,sensors_to_keep=number_of_sensors, mean=synth_mean)
pickle.dump([synth,noisy],open("s-n.p","wb"))

# synth_delta=10
# dimension=10

# synth, noisy = pickle.load(open("s-n.p", "rb"))

x_full = synth



x_train =x_full
x_test = noisy
x_test[x_test<0]= 0
max = np.amax(x_train)
x_train = x_train/max
x_test = x_test/max
normalized_delta = synth_delta/max

x_test = copy_to_shared_Array(x_test,"x_test")

# noisy = noisy / max

# Factorization with MACO
X_u = L_ucalc(x_train, normalized_delta)
X_l = L_ucalc(x_train, -normalized_delta)
X_u = copy_to_shared_Array(X_u, "X_u")
X_l = copy_to_shared_Array(X_l, "X_l")
x_train = copy_to_shared_Array(x_train, "x_train")

RMSE=[]
# # Initial training
L, R = maco(x_train, X_u, X_l, m=0.0001, dimension=dimension, epoch_n=50,RMSEs=RMSE)
# #
pickle.dump([L, R], open("L-R.p", "wb"))
#
# L, R  = pickle.load(open("L-R.p", "rb"))
# L= copy_to_shared_Array(L,"L")
# R= copy_to_shared_Array(R, "R")
# X_u = copy_to_shared_Array(X_u,"X_u")
# X_l = copy_to_shared_Array(X_l,"X_l")
# x_train = copy_to_shared_Array(x_train,"x_train")
# rmse = pickle.load(open("RMSE.p",'rb'))
# RMSE = [rmse] + RMSE


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
epoch_n=1
results=dict()
for delta in range(0, 30, 5):
    results[delta]={}
    results[delta]['positives']=[]
    results[delta]['negatives']=[]

for day in range(0, x_test.shape[0]):
    row = x_test[day, :]
    for delta in range(0,30,5):
        value = solveSystem(R, row, delta=delta/max)
        if value == 0:
            results[delta]['positives']+=[day]
            positives.append(day)
        if value == 2:
            negatives.append(day)
            results[delta]['negatives'] += [day]
    pickle.dump(results, open("results.p", 'wb'))
    x_train[row_to_replace, :] = row

    # Factorization with MACO
    x_u = L_ucalc(row, normalized_delta)
    X_u[day,:]=x_u


    x_l = L_ucalc(row, -normalized_delta)
    X_l[day,:]=x_l

    # Initial training
    L, R = macoUpdate(x_train, X_u, X_l,L,R,m=0.00000, dimension=dimension,
                      epoch_n=epoch_n,rmses=RMSE,color='r')
    row_to_replace += 1
    row_to_replace = row_to_replace % x_train.shape[0]
    print("Row to Replace", row_to_replace)
    print(sp.linalg.norm(x_train))

    rmse_filename= "RMSE_"+str(epoch_n)+".p"
    pickle.dump(RMSE,open(rmse_filename,'wb'))
pickle.dump(positives,open("positives.p",'wb'))
pickle.dump(negatives,open("negatives.p",'wb'))

multiplecoverage(RMSE,str(epoch_n))
# L, R = pickle.load(open("L-R.p","rb"))








#
# for d in range(0, 100):
#     # Initialize local statistic matrices
#     true_positive = []
#     true_negative = []
#     false_positive = []
#     false_negative = []
#     print("Non normalized Delta: ", d)
#     delta = d / max
#     plot_delta = delta
#
#     t0 = time()
#     labels = []
#
#     counter_valid = 0
#     total_avg_distance_valid = 0
#
#     # total_avg_valid=[float]*int(x.shape[1]/daily_samples)
#     total_avg_valid = np.zeros([int(x_train.shape[1] / daily_samples)])
#     counter_invalid = 0
#     total_avg_distance_invalid = 0
#     # total_avg_invalid=[float]*int(x.shape[1]/daily_samples)
#     total_avg_invalid = np.zeros([int(x_train.shape[1] / daily_samples)])
#
#     print("Total Samples", validation_matrix.shape[0])
#     print("Negative Start", negative_start)
#     path = "synthetic-" + str(validation_matrix.shape[0]) + "-" + str(negative_start)
#
#     # Iteration through Matrices samples
#     for l in range(0, int(validation_matrix.shape[0])):
#
#         # Initialize local statistic matrices (for each sample)
#         i = l
#         normal_total = np.zeros([int(x_train.shape[1] / daily_samples), 0])
#         event_total = np.zeros([int(x_train.shape[1] / daily_samples), 0])
#         avg = np.zeros([int(x_train.shape[1] / daily_samples)])
#         avg_distance = 0
#         for k in range(0, int(x_train.shape[1] / daily_samples)):
#             index = np.arange(start=k * daily_samples, stop=k * daily_samples + daily_samples)
#             avg_distance += abs(sensor_avg[k] - np.average(x_train[i, index]))
#             avg[k] = (np.average(x_train[i, index]))
#         value = solveSystem(h=L, b=(validation_matrix[i, :]), delta=delta)
#         if value == 0:
#             if i < negative_start:
#                 true_positive.append(1)
#             if i > negative_start:
#                 false_positive.append(1)
#             total_avg_distance_valid += avg_distance / (x_train.shape[1] / daily_samples)
#             total_avg_valid += avg
#             counter_valid += 1
#             normal_total = np.concatenate([normal_total, avg.reshape((avg.shape[0], 1))], axis=1)
#             new_input_plot(daily_samples=daily_samples, x=x_train[i, :], K=n_components, type="normal", Delta=delta,
#                            day=i, path=path)
#             labels.append(0)
#             daily_avg = []
#             sensor_id = []
#             for y in range(0, int(x_train.shape[1] / daily_samples)):
#                 temp1 = x_train[i, (daily_samples * y):(y * daily_samples + daily_samples)]
#                 temp2 = np.average(temp1)
#                 sensor_id.append(y)
#                 daily_avg.append(temp2)
#             DistanceChart(avgFlow=sensor_avg, normalFlow=[], eventFlow=daily_avg, day=i,
#                           k=n_components, delta=delta, type="normal", path=path)
#         if value == 2:
#
#             if i > negative_start:
#                 true_negative.append(1)
#             if i < negative_start:
#                 false_negative.append(1)
#
#             total_avg_distance_invalid += avg_distance / (x_train.shape[1] / daily_samples)
#             total_avg_invalid += avg
#             event_total = np.concatenate([event_total, avg.reshape((avg.shape[0], 1))], axis=1)
#
#             sensor_id = []
#             daily_avg = []
#             for y in range(0, int(x_train.shape[1] / daily_samples)):
#                 labels.append(1)
#                 temp1 = x_train[i, (daily_samples * y):(y * daily_samples + daily_samples)]
#                 temp2 = np.average(temp1)
#                 sensor_id.append(y)
#                 daily_avg.append(temp2)
#             DistanceChart(avgFlow=sensor_avg, normalFlow=[], eventFlow=daily_avg, day=i,
#                           k=n_components, delta=delta, type="event", path=path)
#             # print(avg/19)
#             labels.append(1)
#             new_input_plot(daily_samples=daily_samples, x=x_train[i, :], K=n_components, type="event", Delta=delta,
#                            day=i,
#                            path=path)
#             counter_invalid += 1
#             # print("EVENT DAY:")
#             # distance = sorted(zip(daily_avg, sensor_id), reverse=True)[:10]
#             # print(distance)
#         # DistanceChart(avgFlow=sensor_avg, normalFlow=normal_total, eventFlow=event_total, day=i, k=n_components)
#         plotStandardDeviation(deviationMatrix=samples_Standard_deviation, avgFlow=sensor_avg,
#                               normalFlow=normal_total, eventFlow=event_total, day=i, k=n_components,
#                               type="Standard", delta=delta, path=path)
#         plotStandardDeviation(deviationMatrix=samples_Mean_deviation, avgFlow=sensor_avg, normalFlow=normal_total,
#                               eventFlow=event_total, day=i, k=n_components, type="Mean", delta=delta, path=path)
#     print("--------------------------------")
#     print("Delta: ", delta)
#     print("-----")
#     print("Normal: ", (counter_valid / (counter_invalid + counter_valid)) * 100, "%")
#     plot_valid_percentage.append(counter_valid / (x_train.shape[0] / step_n) * 100)
#     if counter_valid != 0:
#         print("Distance from average: ", total_avg_distance_valid / counter_valid)
#     else:
#         print("Distance from average: Not Found")
#
#     print("-----")
#     print("Event : ", (counter_invalid / (counter_invalid + counter_valid)) * 100, "%")
#     plot_invalid_percentage.append((counter_invalid / (x_train.shape[0] / step_n)) * 100)
#     if counter_invalid != 0:
#         print("Distance from average: ", total_avg_distance_invalid / counter_invalid)
#     else:
#         print("Distance from average:Not Found ", )
#
#     print("--------------------------------")
#     positive.append((counter_valid / (x_train.shape[0] / step_n)) * 100)
#     negative.append((counter_invalid / (x_train.shape[0] / step_n)) * 100)
#     deltas_total.append(delta)
#     if counter_valid != 0:
#         normal_flow.append(total_avg_distance_valid / counter_valid)
#         # total_avg_valid = [x / counter_valid for x in total_avg_valid]
#         total_avg_valid = total_avg_valid / counter_valid
#         deltas_valid.append(delta)
#     if counter_invalid != 0:
#         event_flow.append(total_avg_distance_invalid / counter_invalid)
#         deltas_invalid.append(delta)
#         total_avg_invalid = total_avg_invalid / counter_invalid
#         # total_avg_invalid = [x / counter_invalid for x in total_avg_invalid]
#
#     avgDistanceChart(avgFlow=sensor_avg, normalFlow=total_avg_valid, eventFlow=total_avg_invalid, delta=delta,
#                      k=n_components, path=path)
#     # new_input_whole_sensor_data_plot(daily_samples=daily_samples, x=x, K=n_components, Delta=delta,
#     #                                  remained=remained, labels=s, path=path)
#     print("Test done in %0.3fs." % (time() - t0))
#     print("True positives", len(true_positive))
#     print("True negatives", len(true_negative))
#     print("False positives", len(false_positive))
#     print("False negatives", len(false_negative))
#     precision = len(true_positive) / (len(true_positive) + len(false_positive))
#     recall = len(true_positive) / (len(true_positive) + len(false_negative))
#     # F-Measure
#     # b=0.1
#     # f_measure = (1+b*b)*((precision*recall)/(b*b*precision+recall))
#
#     f1_score = 2 * (precision * recall) / (precision + recall)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1 Score:", f1_score)
#     recall_list.append(recall)
#     precision_list.append(precision)
#     f1_score_list.append(f1_score)
#     delta_list.append(d)
#
# # Varius Plots
#
# plotValidInvalidPercentage(k=n_components, valid=plot_valid_percentage, invalid=plot_invalid_percentage,
#                            deltas=deltas_total, path=path)
# deltaVarianceChart(deltas_total, positive, negative, k=n_components, path=path)
# deltaVarianceImpactOnFlowChart(deltas_valid, deltas_invalid, normal_flow, event_flow, k=n_components, path=path)
# recallPlot(recall_list, delta_list, path=path)
# precisionPlot(precision_list, delta_list, path=path)
# f1ScorePlot(f1_score_list, delta_list, path=path)
