from theano.tensor.shared_randomstreams import RandomStreams

__author__ = 'quynhdo'
import theano as th
import theano.tensor as T
import numpy as np

import theano.sparse
import scipy
import theano.sparse.basic as ST
#### transfer functions

def DotTransferFunction(x, W, b):
        if b !=None:
            return th.dot(x, W) + b
        else:
            return th.dot(x,W)

def SpDotTransferFunction(x, W, b):
        if b !=None:
            return ST.dot(x, W) + b
        else:
            return ST.dot(x,W)

def DotTransferFunctionExtended( x, x_e, W, b):

        if b != None:
        #    return th.dot(T.concatenate([x,x_e], axis=1), W) + b
            if isinstance(x_e, (scipy.sparse.spmatrix, np.ndarray, tuple, list)):
                if theano.sparse.basic._is_sparse(x_e):
                    xx = th.sparse.hstack((theano.sparse.csr_from_dense(x),x_e))
                    return th.sparse.structured_dot(xx, W) + b
            else:
                if theano.sparse.basic._is_sparse_variable(x_e):
                    xx = th.sparse.hstack((theano.sparse.csr_from_dense(x),x_e))
                    return th.sparse.structured_dot(xx, W) + b

            return th.dot(T.concatenate([x,x_e], axis=1), W) + b

        else:
            return T.dot(x,W)

def NoneTransferFunction(self, x, W, b):
        return x



#### activate functions
SigmoidActivateFunction = T.nnet.sigmoid
SoftmaxActivateFunction = T.nnet.softmax
TanhActivateFunction = T.tanh
SpSigmoidActivateFunction = ST.structured_sigmoid
def NoneActivateFunction (x):
    return x
def RectifierActivateFunction(x):
			return x*(x>0)

#### cost functions
def NegativeLogLikelihoodCostFunction(o, y):
    '''
    Used for Vector output
    :param o: output of the system
    :param y: gold
    :return:
    '''
    return -T.mean(T.log(o)[T.arange(y.shape[0]), y])



def SquaredErrorCostFunction(o,y):
    '''
    used for matrix output
    :param o:
    :param y:
    :return:
    '''
    return T.mean((o-y) ** 2)

def CrossEntropyCostFunction(o, y):
    '''
    used for matrix output
    :param o:
    :param y:
    :return:
    '''
    L = - T.sum(y * T.log(o) + (1 - y) * T.log(1 - o), axis=1)

    cost = T.mean(L)
    return cost

def SpCrossEntropyCostFunction(o, y):
    '''
    used for matrix output
    :param o:
    :param y:
    :return:
    '''
    L = - ST.sp_sum(y * ST.structured_log(o) + (1 - y) * ST.structured_log(1 - o), axis=1)

    cost = T.mean(L)
    return cost


#### functions to process input for the input layer

def NoneProcessFunction(x, *args): return x



#### init weights

def WeightInit(size_in, size_out, lamda=1, rng=None):
     #rng = np.random.RandomState(89677)
     if rng is None:
        rng = np.random.RandomState(123)
     theano_rng = RandomStreams(rng.randint(2 ** 30))
     return lamda * rng.uniform(
                            low=-1.0,
                            high=1.0,
                            size=(size_in , size_out)
                        )
