from nn.Layer import Layer
from nn.Connection import Connection
from nn.Functions import DotTransferFunction, SigmoidActivateFunction
import theano.tensor as T
import theano as th
import timeit
import numpy as np
import pickle
import os
__author__ = 'quynhdo'


class NNNet:
    '''
    this class is used to implement a neural neutron network
    '''
    Output_Type_Real="real"
    Output_Type_Binary="binary"
    Output_Type_SoftMax="softmax"


    def __init__(self, costfunction, idx="", otype="real", x=None, x_type="matrix", y=None):   # x is a list to support a net that has more than 1 input layer
        '''

        :param costfunction: cost function
        :param idx: id of the network if necessary
        :param otype: type of the output which can be either real (the real output),
         binary (the output values are rounded to 0 or 1) or softmax (as labels - using argmax)
        :param x: training input
        :param x_type: either matrix, vector or sparse matrix
        :param y: training output
        :return:
        '''
        self.layers = []
        self.connections = []
        self.cost_function =  costfunction
        self.otype = otype
        self.idx = idx
        self.input_layers = [] # input layers that need input values
        if x is None:
            if x_type == "matrix":
                self.x = T.matrix(name=self.idx + "_x", dtype=th.config.floatX)
            if x_type == "vector":
                self.x = T.vector(name=self.idx + "_x", dtype=th.config.floatX)
            if x_type == "sparse":
                self.x = th.sparse.csc_matrix(name=self.idx + "_x", dtype=th.config.floatX)
        else:
            self.x = x
        if y is None:
            if otype == "sparse":
                self.y = th.sparse.csc_matrix(name=self.idx + "_y", dtype=th.config.floatX)
            if otype == "real":
                self.y = T.matrix(name=self.idx + "_y", dtype=th.config.floatX)
            else:
                self.y = T.ivector(name=self.idx + "_y")
        else:
            self.y = y
        self.output_layer = None
        self.params = []
        self.y_pred = None   # predicted output values
        self.p_y_given_x = None   # real outcomes of the network


    def addLayer(self, l):
        '''
        add one layer to the network
        :param l:
        :return:
        '''
        if isinstance(l, list):
            for ll in l:
                self.addLayer(ll)
        else:
            l.idx = self.idx + "_" + "l" + str(len(self.layers))
            self.layers.append(l)
            if l.ltype == Layer.Layer_Type_Input:
                self.input_layers.append(l)
            if l.ltype == Layer.Layer_Type_Output:
                self.output_layer = l

    def addConnection(self, conn):
        '''
        add connection to the network
        :param conn:
        :return:
        '''
        if isinstance(conn, list):
            self.connections += conn
        else:
            self.connections.append(conn)

    def createConnection(self, scr, dst, transfer_func=DotTransferFunction, activate_func=SigmoidActivateFunction, use_bias=True,
                 initial_w=None, initial_b=None,
                 w_lamda=1.0, rng=None):
        '''
        create a connection
        :param scr: source layer
        :param dst: target layer
        :param transfer_func: transfer function
        :param activate_func: activate function
        :param use_bias: use bias or not
        :param initial_w: initial weights
        :param initial_b: initial bias
        :param w_lamda: w_lamda used to initiate weights value
        :param rng:
        :return:
        '''
        cidx = self.idx + "_" + "c" + str(len(self.connections))
        conn = Connection(scr, dst, transfer_func, activate_func, use_bias, cidx, initial_w, initial_b, w_lamda, rng)
        self.connections.append(conn)
        for p in conn.params:
            self.params.append(p)
        return conn

    def start(self, inputs=None):
        '''
        start the network, data is transfered through the network
        :param inputs:
        :return:
        '''
        if inputs is None:
            self.input_layers[0].process_input(self.x)
        else:
            for i in range(len(self.input_layers)):
                self.input_layers[i].process_input(inputs[i])
        for conn in self.connections:
            conn.start()
        self.p_y_given_x = self.output_layer.units
        self.y_pred = self.computeOutput(self.p_y_given_x)

    def computeOutput(self, y):
        '''
        compute output
        :param y:
        :return:
        '''
        if self.otype == NNNet.Output_Type_Binary:
            return T.round(y)

        if self.otype == NNNet.Output_Type_SoftMax:
            return T.argmax(y, axis=1)

        return y

    def errors(self, y):
        '''
        get error
        :param y:
        :return:
        '''
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def get_cost_updates(self,learning_rate, regularization_name=None, regularization_lambda=0.0001):
        '''
        get the cost update function
        :param learning_rate:
        :param regularization_name:
        :param regularization_lambda:
        :return:
        '''
        cost = self.cost_function(self.p_y_given_x, self.y)

        if regularization_name is not None:
            L=0
            if regularization_name == "l1":
                for conn in self.connections:
                    L += abs(conn.W).sum()

            if regularization_name == "l2":
                for conn in self.connections:
                    L += abs(conn.W ** 2).sum()

            cost += regularization_lambda * L

        gparams = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return cost, updates



    def predict(self, test_data):
        '''
        predict
        :param test_data:
        :return:
        '''
        test_model = th.function(
                inputs=[],
                outputs=self.y_pred,
                givens={
                    self.x: test_data ,

                }
            )

        return test_model()

    def evaluate(self, test_data, test_data_label):
        evaluate_model = th.function(
                inputs=[],
                outputs=self.errors(self.y),
                givens={
                    self.x: test_data,
                    self.y : test_data_label


                }
            )

        return evaluate_model()

    def getHidden(self, x):
        '''
        get output of the top hidden layer
        :param x:
        :return:
        '''
        t = x
        for i in range(len(self.connections)-1):
            t = self.connections[i].getOutput(t)
        return t

    def fit(self, train_data, train_data_label, batch_size, training_epochs, learning_rate, validation_data=None, validation_data_label=None,
            save_model_path= None, regularization_name=None, regularization_lambda=0.0001):
        '''
        Default fit function for one input: x's dim = 1
        :param train_data:
        :param train_data_label:
        :param batch_size:
        :param training_epochs:
        :param learning_rate:
        :param validation_data:
        :param validation_data_label:
        :param save_model_path:
        :param regularization_name:
        :param regularization_lambda:
        :return:
        '''
        index = T.lscalar()
        # models for training
        cost,updates=self.get_cost_updates(learning_rate, regularization_name,regularization_lambda)

        train_model = th.function(
        [index],
        cost,
        updates=updates,
        givens={
            self.x: train_data[index * batch_size: (index + 1) * batch_size],
            self.y: train_data_label[index * batch_size: (index + 1) * batch_size]

            }
        )

        # models for validating
        validate_model=None
        n_train_batches = (int) (train_data.get_value(borrow=True).shape[0] / batch_size)
        n_valid_batches = -1
        if validation_data is not None:

            validate_model = th.function(
                inputs=[index],
                outputs=self.errors(self.y),
                givens={
                    self.x: validation_data[index * batch_size: (index + 1) * batch_size],
                    self.y: validation_data_label[index * batch_size:(index + 1) * batch_size]

                }
            )

            n_valid_batches = (int)(validation_data_label.get_value(borrow=True).shape[0] / batch_size)

     #   n_train_batches =2
        start_time = timeit.default_timer()
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        done_looping = False
        patience_increase = 2  # wait this much longer when a new best is
                           # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
        patience = 1000000
        validation_frequency = min(n_train_batches, patience / 2)

        epoch=0
        while (epoch < training_epochs) and (not done_looping):
            epoch = epoch + 1
            c = []
            for minibatch_index in range(int(n_train_batches)):

                minibatch_avg_cost = train_model(minibatch_index)
                c.append(minibatch_avg_cost)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index
                if validation_data is not None:
                    if (iter + 1) % validation_frequency == 0:


                        # compute zero-one loss on validation set
                        validation_losses = [validate_model(i) for i
                                             in range(int(n_valid_batches))]
                        this_validation_loss = np.mean(validation_losses)

                        print(
                            'epoch %i, minibatch %i/%i, validation error %f %%' %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                this_validation_loss * 100.
                            )
                        )

                        # if we got the best validation score until now
                        if this_validation_loss < best_validation_loss:
                            #improve patience if loss improvement is good enough
                            if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                            ):
                                patience = max(patience, iter * patience_increase)

                            best_validation_loss = this_validation_loss
                            best_iter = iter


                            if save_model_path is not None:
                                with open(save_model_path, 'wb') as f:
                                    pickle.dump(self, f)



                if patience <= iter:
                    done_looping = True
                    break
            print ('Training epoch %d, cost ' % epoch, np.mean(c))
        end_time = timeit.default_timer()
        if validation_data is not None:
            print(('Optimization complete. Best validation score of %f %% '
                   'obtained at iteration %i, with test performance %f %%') %
                  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))


