from nn.Functions import DotTransferFunction, SigmoidActivateFunction, WeightInit
import theano.tensor as T
import theano as th
import numpy as np
__author__ = 'quynhdo'


# implement the connection between two layers


class Connection:


    def __init__(self, scr, dst, transfer_func=DotTransferFunction, activate_func=SigmoidActivateFunction, use_bias=True,
                 idx="", initial_w=None, initial_b=None,
                 w_lamda=1.0, rng = None):
        '''

        :param scr: source layers, can be a single layer of a list of layer, if is a list of layer then we have to concatenate the units
        of the source layers to process in the connection
        :param dst: destination layer
        :param transfer_func: usually the Dot func to get the net input
        :param activate_func: activate function
        :param use_bias:
        :param idx: index of the connnection
        :param initial_w: initial value for weights
        :param initial_b: initial value for bias
        :param w_lamda: initial value to init the weight
        :return:
        '''

        self.scr = scr  # source layer
        self.dst = dst  # destination layer
        self.activate_func = activate_func # transfer function
        self.transfer_func = transfer_func  # activate function
        self.W = None
        self.b = None
        if isinstance(self.scr, list): # if there are more than one src layers
            self.size_in = np.sum([l.size for l in self.scr])  # then input size of the connection equals to the sum of all scr layers
        else:
            self.size_in = self.scr.size

        self.size_out = self.dst.size

        if initial_w is not None:
            self.W = initial_w
        else:
            self.W = th.shared(value=np.asarray(WeightInit(self.size_in, self.size_out, w_lamda, rng), dtype=th.config.floatX)
                    , name='W'+idx, borrow=True)
        self.params = [self.W]
        if use_bias:
            if initial_b is not None:
                self.b = initial_b
            else:
                self.b = th.shared(value=np.zeros(dst.size, dtype=th.config.floatX), name="b" + idx, borrow=True )
            self.params.append(self.b)

    # Start the connection,  calculate the unit values of the dst layer from the scr layer
    def start(self):

        xx = None
        start_sparse_idx = -1
        if isinstance(self.scr, list): # we only allow sparse layers to occur at the end...
            for i in range(len(self.scr)):
                if th.sparse.basic._is_sparse_variable(self.scr[i].units):
                    start_sparse_idx = i
                    break
            if start_sparse_idx > 0:
                xx = T.concatenate([self.scr[i].units for i in range (start_sparse_idx)], axis=1)
                xx = th.sparse.hstack((th.sparse.csr_from_dense(xx),self.scr[start_sparse_idx].units))
                for j in range(start_sparse_idx+1,len(self.scr)):
                    xx = th.sparse.hstack(xx, self.scr[j].units)
            if start_sparse_idx == 0:
                xx =  self.scr[0].units
                for j in range(1,len(self.scr)):
                    xx = th.sparse.hstack(xx, self.scr[j].units)

            if start_sparse_idx < 0:
                xx = T.concatenate([self.scr[i].units for i in range (len(self.scr))], axis=1)
        else:
            xx = self.scr.units

        self.dst.units = self.activate_func(self.transfer_func(xx, self.W, self.b))


    def getOutput(self, x):
        '''
        only work when x is a single varibale, not a list
        :param x:
        :return:
        '''
        return self.activate_func(self.transfer_func(x, self.W, self.b))
