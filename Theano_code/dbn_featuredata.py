import os
import sys
import time

import numpy
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from data_process import loadFeaturedData

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from rbm import RBM
from dbn import DBN

def test_DBN(finetune_lr=0.01, pretraining_epochs=100,
             pretrain_lr=0.1, k=1, training_epochs=100,
             batch_size=5):

    datasets = loadFeaturedData()

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)

    print '... building the model'

    dbn = DBN(numpy_rng=numpy_rng, n_ins=10,
              hidden_layers_sizes=[100],
              n_outs=2)

    #########################
    # PRETRAINING THE MODEL #
    #########################

    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = time.clock()
    # Pre-train layer-wise
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = time.clock()
    # end-snippet-2
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, test_model, test_confmatrix = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    index = T.lscalar('index')  # index to a [mini]batch

    # Custom function
    stopcheck_model = theano.function(
        inputs=[index],
        outputs=dbn.finetune_cost,
        givens={
            dbn.x: test_set_x[index * batch_size:(index + 1) * batch_size],
            dbn.y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    train_check = theano.function(
        inputs=[index],
        outputs=dbn.errors,
        givens={
            dbn.x: train_set_x[index * batch_size: (index + 1) * batch_size],
            dbn.y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print '... finetuning the model'

    test_score = 0.
    start_time = time.clock()

    tr_cost = []
    te_cost = []

    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        # go through the training set
        d = []
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            d.append(minibatch_avg_cost)
        print(('In epoch %d, ') % (epoch))
        print 'Training cost = ', numpy.mean(d)
        tr_cost.append(numpy.mean(d))

        tc = []
        for test_batch_index in xrange(n_test_batches):
            testing_cost = stopcheck_model(test_batch_index)
            tc.append(testing_cost)
        print 'Testing cost = ', numpy.mean(tc)
        te_cost.append(numpy.mean(tc))


    train_losses = [train_check(i) for i in xrange(n_train_batches)]
    print 'Training error = ', numpy.mean(train_losses)*100, ' %'

    test_losses = test_model()
    test_score = numpy.mean(test_losses)

    test_confmatrices = test_confmatrix()
    test_confelement = numpy.sum(test_confmatrices, axis=0)
    true_pos = test_confelement[0]
    true_neg = test_confelement[1]
    false_pos = test_confelement[2]
    false_neg = test_confelement[3]
    f_score = (true_pos + true_neg)/float(true_pos + true_neg + false_pos + 5*false_neg)

    print(('Test error: %f %%, F-score: %f') % (test_score * 100., f_score))
    print(('TP %i, TN %i, FP %i, FN %i') % (true_pos, true_neg, false_pos, false_neg))

    end_time = time.clock()
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))

    x_axis = numpy.arange(training_epochs)
    plt.plot(x_axis, numpy.array(tr_cost), '+', x_axis, numpy.array(te_cost), '.')
    plt.show()

if __name__ == '__main__':
    test_DBN()
