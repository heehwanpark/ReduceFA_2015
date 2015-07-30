import numpy
import random
import pickle
import sys
import time

from data_process import shared_dataset
from mlp_hh import test_mlp

def load10FeatureData(input_data, target_data, shuffled_index, test_idx):
    if test_idx == 0:
        testIdx = shuffled_index[:75]
        trainIdx = shuffled_index[75:]
    elif test_idx == 9:
        testIdx = shuffled_index[675:]
        trainIdx = shuffled_index[:675]
    else :
        testIdx = shuffled_index[75*test_idx:75*(test_idx+1)]
        trainIdx = shuffled_index[:75*(test_idx)]+shuffled_index[75*(test_idx+1):]

    XtrainSet = [input_data[i] for i in trainIdx]
    YtrainSet = [target_data[j] for j in trainIdx]
    train_set = (XtrainSet, YtrainSet)

    XtestSet = [input_data[k] for k in testIdx]
    YtestSet = [target_data[l] for l in testIdx]
    test_set = (XtestSet, YtestSet)

    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__':

    # final Test
    arch = [400, 400, 400]
    reg = [0., 0.]
    batch_size = 5
    epoch = 4700

    # Make grid
    print '... making condition'

    # Load 10 feature Data
    print '... loading data'

    # Ubuntu
    input_data = numpy.loadtxt(open('/home/heehwan/Documents/workspace/data/DBN_data/X_1405_10features.csv', 'rb'), delimiter=',')
    target_data = numpy.loadtxt(open('/home/heehwan/Documents/workspace/data/DBN_data/Y_1405_10features.csv', 'rb'), delimiter=',')

    # Make dataset index
    idx = range(750)
    random.seed(1)
    random.shuffle(idx)

    print '... starting experiments'
    start_time = time.clock()

    # 10-fold
    result_array = numpy.zeros((10, 4))
    for i in xrange(10):
        print (('Experiment %d/10 is on-going') % (i+1))
        datasets = load10FeatureData(input_data, target_data, idx, i)
        # test_result = [training_accuracy, f-score, testing_accuracy, f-score]
        test_result = test_mlp(architecture = arch,
                                reg_rate = reg,
                                batch_size = batch_size,
                                epoch_num = epoch,
                                datasets = datasets)
        result_array[i,:] = numpy.array(test_result)

    end_time = time.clock()
    print (('It takes for %.2fm') % ((end_time - start_time)/60.))
    # Result
    pickle.dump(result_array, open('result_400_3_v1.pkl', 'wb'))
    print result_array
