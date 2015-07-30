import numpy
import scipy
import random
import theano
import theano.tensor as T

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def loadFeaturedData() :
    print ('... loading data')

    # Windows
    # input = numpy.loadtxt(open('C:\Users\heehwan\Documents\MATLAB\AI&DM\Project\DBN_data\X_1405_10features.csv', 'rb'), delimiter=',')
    # target = numpy.loadtxt(open('C:\Users\heehwan\Documents\MATLAB\AI&DM\Project\DBN_data\Y_1405_10features.csv', 'rb'), delimiter=',')

    # Ubuntu
    input = numpy.loadtxt(open('/home/heehwan/Documents/workspace/data/DBN_data/X_1405_10features.csv', 'rb'), delimiter=',')
    target = numpy.loadtxt(open('/home/heehwan/Documents/workspace/data/DBN_data/Y_1405_10features.csv', 'rb'), delimiter=',')

    idx = range(750)
    random.seed(1)
    random.shuffle(idx)
    trainIdx = idx[:675]
    testIdx = idx[675:]

    XtrainSet = [input[i] for i in trainIdx]
    YtrainSet = [target[j] for j in trainIdx]
    train_set = (XtrainSet, YtrainSet)

    XtestSet = [input[k] for k in testIdx]
    YtestSet = [target[l] for l in testIdx]
    test_set = (XtestSet, YtestSet)

    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval

def load10secData():
    print ('... loading data')

    # Windows
    input = numpy.loadtxt(open('C:\Users\heehwan\Documents\MATLAB\AI&DM\Project\DBN_data\data_10sec_input.dat', 'rb'), delimiter=',')
    target = numpy.loadtxt(open('C:\Users\heehwan\Documents\MATLAB\AI&DM\Project\DBN_data\data_10sec_target.dat', 'rb'), delimiter=',')

    # training set: 14400 validation set: 3600,  test set: 4500

    trainIdx = range(0, 14400)
    validIdx = range(14400, 18000)
    testIdx = range(18000, 22500)

    XtrainSet = [input[i] for i in trainIdx]
    YtrainSet = [target[j] for j in trainIdx]
    train_set = (XtrainSet, YtrainSet)

    XtestSet = [input[k] for k in testIdx]
    YtestSet = [target[l] for l in testIdx]
    test_set = (XtestSet, YtestSet)

    XvalidSet = [input[n] for n in validIdx]
    YvalidSet = [target[m] for m in validIdx]
    valid_set = (XvalidSet, YvalidSet)

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def load10sec_ECGII_data():
    print ('... loading data')

    # Normal ECG
    # pretraining = numpy.loadtxt(open('C:\Users\heehwan\Documents\MATLAB\AI&DM\Project\DBN_data\data_10sec_dbn_ECGII_pretraining.csv', 'rb'), delimiter=',')
    # training_input = numpy.loadtxt(open('C:\Users\heehwan\Documents\MATLAB\AI&DM\Project\DBN_data\data_10sec_dbn_ECGII_training.csv', 'rb'), delimiter=',')
    # training_target = numpy.loadtxt(open('C:\Users\heehwan\Documents\MATLAB\AI&DM\Project\DBN_data\data_10sec_dbn_ECGII_target.csv', 'rb'), delimiter=',')

    # Wavelet detail
    # pretraining = numpy.loadtxt(open('C:\Users\heehwan\Documents\MATLAB\AI&DM\Project\DBN_data\data_10sec_dbn_ECGII_pretraining_swd.csv', 'rb'), delimiter=',')
    # training_input = numpy.loadtxt(open('C:\Users\heehwan\Documents\MATLAB\AI&DM\Project\DBN_data\data_10sec_dbn_ECGII_training_swd.csv', 'rb'), delimiter=',')
    # training_target = numpy.loadtxt(open('C:\Users\heehwan\Documents\MATLAB\AI&DM\Project\DBN_data\data_10sec_dbn_ECGII_target.csv', 'rb'), delimiter=',')

    # Ubuntu - Wavelet detail
    pretraining = numpy.loadtxt(open('/home/heehwan/Documents/workspace/data/DBN_data/data_10sec_dbn_ECGII_pretraining_nan.csv', 'rb'), delimiter=',')
    training_input = numpy.loadtxt(open('/home/heehwan/Documents/workspace/data/DBN_data/data_10sec_dbn_ECGII_training_nan.csv', 'rb'), delimiter=',')
    training_target = numpy.loadtxt(open('/home/heehwan/Documents/workspace/data/DBN_data/data_10sec_dbn_ECGII_target_nan.csv', 'rb'), delimiter=',')

    idx = range(728)
    random.seed(123)
    random.shuffle(idx)
    trainIdx = idx[:600]
    testIdx = idx[600:]

    XtrainSet = [training_input[i] for i in trainIdx]
    YtrainSet = [training_target[j] for j in trainIdx]
    train_set = (XtrainSet, YtrainSet)

    XtestSet = [training_input[k] for k in testIdx]
    YtestSet = [training_target[l] for l in testIdx]
    test_set = (XtestSet, YtestSet)

    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    pretraining_x = theano.shared(numpy.asarray(pretraining,
                                           dtype=theano.config.floatX),
                             borrow=True)

    rval = [pretraining_x, (train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval
