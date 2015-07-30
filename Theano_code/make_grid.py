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
    arch_list = ([400, 400, 400], [500, 500, 500])
    reg = [0., 0.]
    batch_size = 5
    epoch = 5000

    # # 21 Test cases
    # arch_list = ([200, 200, 200], [300, 300, 300], [400, 400, 400], [200, 200, 200, 200])
    # reg = [0., 0.]
    # learning_rate = 0.001
    # batch_size = 5
    # epoch = 5000

    # 20 Test cases
    # arch_list = ([100, 100], [200, 200], [300, 300],
    #             [100, 100, 100], [200, 200, 200], [300, 300, 300])
    # reg = [0., 0.]
    # learning_rate = 0.001
    # batch_size = 5
    # epoch_list = (2000, 3000)3

    # 18-20 Test cases
    # arch_list = ([20], [50], [100], [200],
    # [20, 20], [50, 50], [100, 100], [200, 200],
    # [50, 50, 50], [100, 100, 100])
    # reg_list = ([0., 0.], [0.001, 0.], [0., 0.001])
    # learning_rate_list = (0.01, 0.001)
    # batch_size_list = (5, 10)


    # Make grid
    print '... making condition grid'

    cond_list = []
    for arch in arch_list:
        cond_list.append((arch, reg, batch_size, epoch))

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

    cond_result_list = []
    for condition in cond_list:
        start_time = time.clock()
        print '#####################'
        print 'Current condition: '
        print '- hidden architecture', condition[0]
        print '- regularization rate', condition[1]
        print '- batch size', condition[2]
        print '- Epoch number', condition[3]

        # 10-fold
        result_array = numpy.zeros((10, 3))
        for i in xrange(10):
            print (('Experiment %d/10 is on-going') % (i+1))
            datasets = load10FeatureData(input_data, target_data, idx, i)
            # test_result = [training_accuracy, testing_accuracy, f-score]
            test_result = test_mlp(architecture = condition[0],
                                    reg_rate = condition[1],
                                    batch_size = condition[2],
                                    epoch_num = condition[3],
                                    datasets = datasets)
            result_array[i,:] = numpy.array(test_result)

        result_mean = numpy.mean(result_array, axis=0)
        result_std = numpy.std(result_array, axis=0)
        result = numpy.concatenate((result_mean, result_std), axis=1)
        cond_result_list.append(result)

        print 'Result: '
        print (('Mean training accuracy %f %%, testing accuracy %f %%, f-score %f') % (result_mean[0], result_mean[1], result_mean[2]))
        end_time = time.clock()
        print (('It takes for %.2fm') % ((end_time - start_time)/60.))

    # Output cond_result
    pickle.dump(cond_list, open('condition_list_v4.pkl', 'wb'))
    pickle.dump(cond_result_list, open('condition_result_v4.pkl', 'wb'))
