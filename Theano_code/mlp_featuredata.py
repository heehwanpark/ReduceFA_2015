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
from mlp_hh import MLP

if __name__ == '__main__':

    datasets = loadFeaturedData()

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    batch_size = 5

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(123)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=10,
        hidden_layers_sizes=[400, 400, 400],
        n_out=2
    )

    L1_reg = 0.
    L2_reg = 0.

    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    test_confmatrix = theano.function(
        inputs=[index],
        outputs=classifier.confusion_matrix(y),
        givens={
            x: test_set_x[
                index * batch_size: (index + 1) * batch_size
            ],
            y: test_set_y[
                index * batch_size: (index + 1) * batch_size
            ]
        }
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    # learning_rate = 0.001
    l_r = T.scalar('l_r', dtype=theano.config.floatX)
    updates = []
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - gparam * l_r))

    train_model = theano.function(
        inputs=[index, l_r],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    stopcheck_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    train_check = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    ###############
    # TRAIN MODEL #
    ###############
    start_time = time.clock()

    tr_cost = []
    te_cost = []

    training_epochs = 5000
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        learning_rate = 0.001/(1+0.001*epoch)
        # go through the training set
        d = []
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index, learning_rate)
            d.append(minibatch_avg_cost)
        print(('In epoch %d,') % (epoch))
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

    test_losses = [test_model(i) for i in xrange(n_test_batches)]
    test_score = numpy.mean(test_losses)

    test_confmatrices = [test_confmatrix(i) for i in xrange(n_test_batches)]
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
