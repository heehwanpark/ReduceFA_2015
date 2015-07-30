# MLP class modified by heehwan

import time
import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """
    MLP class modified by heehwan. It supports deep architecture.
    """
    def __init__(self, rng, input, n_in, hidden_layers_sizes, n_out):

        self.hidden_layers = []
        self.n_layers = len(hidden_layers_sizes)
        self.params = []

        w_sum = 0
        w_square_sum = 0

        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = input
            else:
                layer_input = self.hidden_layers[i - 1].output

            sigmoid_layer = HiddenLayer(rng=rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.tanh)

            self.hidden_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            w_sum = w_sum + abs(sigmoid_layer.W).sum()
            w_square_sum = w_square_sum + (sigmoid_layer.W ** 2).sum()

        self.logRegressionLayer = LogisticRegression(
            input=self.hidden_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_out
        )
        self.params.extend(self.logRegressionLayer.params)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small

        self.L1 = (
            w_sum + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            w_square_sum + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # Compute Confusion_matrix by heehwan
        self.confusion_matrix = self.logRegressionLayer.confusion_matrix

def test_mlp(architecture, reg_rate, batch_size, epoch_num, datasets):

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

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
        hidden_layers_sizes=architecture,
        n_out=2
    )

    L1_reg = reg_rate[0]
    L2_reg = reg_rate[1]

    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # Result function
    train_error = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_confmatrix = theano.function(
        inputs=[index],
        outputs=classifier.confusion_matrix(y),
        givens={
            x: train_set_x[
                index * batch_size: (index + 1) * batch_size
            ],
            y: train_set_y[
                index * batch_size: (index + 1) * batch_size
            ]
        }
    )

    test_error = theano.function(
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

    # Gradient of parameters
    gparams = [T.grad(cost, param) for param in classifier.params]

    l_r = T.scalar('l_r', dtype=theano.config.floatX)
    # Update fomular
    updates = [
        (param, param - l_r * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # Training function
    train_model = theano.function(
        inputs=[index, l_r],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Start training
    training_epochs = epoch_num
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        learning_rate = 0.001/(1+0.001*epoch)
        # go through the training set
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index, learning_rate)

    # Result
    train_losses = [train_error(i) for i in xrange(n_train_batches)]
    train_accuracy = 1 - numpy.mean(train_losses)

    train_confmatrices = [train_confmatrix(i) for i in xrange(n_train_batches)]
    train_confelement = numpy.sum(train_confmatrices, axis=0)
    tr_true_pos = train_confelement[0]
    tr_true_neg = train_confelement[1]
    tr_false_pos = train_confelement[2]
    tr_false_neg = train_confelement[3]
    tr_f_score = (tr_true_pos + tr_true_neg)/float(tr_true_pos + tr_true_neg + tr_false_pos + 5*tr_false_neg)


    test_losses = [test_error(i) for i in xrange(n_test_batches)]
    test_accuracy = 1 - numpy.mean(test_losses)

    test_confmatrices = [test_confmatrix(i) for i in xrange(n_test_batches)]
    test_confelement = numpy.sum(test_confmatrices, axis=0)
    te_true_pos = test_confelement[0]
    te_true_neg = test_confelement[1]
    te_false_pos = test_confelement[2]
    te_false_neg = test_confelement[3]
    te_f_score = (te_true_pos + te_true_neg)/float(te_true_pos + te_true_neg + te_false_pos + 5*te_false_neg)

    return [train_accuracy, tr_f_score, test_accuracy, te_f_score]
