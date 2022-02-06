import numpy as np
import collections

def calculateDistribution(x):


    max_value = np.max(x)
    # print(collections.Counter(x.flatten()))

    return collections.OrderedDict(sorted(dict(collections.Counter(x.flatten())).items()))
    # for i in range(0,max_value+1):
    #     temp=
