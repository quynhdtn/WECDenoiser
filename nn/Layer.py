from theano.tensor.shared_randomstreams import RandomStreams
from nn.Functions import NoneProcessFunction

__author__ = 'quynhdo'
import numpy as np
import theano as th
import  theano.tensor as T

class Layer:

    Layer_Type_Input = "input"
    Layer_Type_Hidden = "hidden"
    Layer_Type_Output = "output"

    def __init__(self, numNodes, ltype, idx="", input_process_func=NoneProcessFunction):
        self.ltype = ltype
        self.units = np.zeros(numNodes, dtype=th.config.floatX)
        self.size = numNodes
        self.id = idx
        self.input_process_func = input_process_func
        self.isExtended=False

    def process_input(self, x):
        self.units = self.input_process_func(x)


class DenoisingLayer(Layer):
    def __init__(self, numNodes, ltype, useBias=True, idx="",  corruption_level=0.1, rng=None, theano_rng=None):
        Layer.__init__(self, numNodes, ltype, useBias, idx)
        self.corruption_level = corruption_level
        if rng is not None:
            self.rng=rng
        else:
            self.rng = np.random.RandomState(123456)
        if theano_rng is not None:
            self.theano_rng=theano_rng
        else:
            self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))
        self.input_process_func = self.denoising_process_function

    def denoising_process_function(self, x):
        return self.theano_rng.binomial(size=x.shape, n=1, p=1 - self.corruption_level, dtype=th.config.floatX) * x
    #    xx = self.gaussian_perturb(x, self.corruption_level)
    #    return xx



    def gaussian_perturb(self,arr, std):
        """Return a Theano variable which is perturbed by additive zero-centred
        Gaussian noise with standard deviation ``std``.
        Parameters
        ----------
        arr : Theano variable
            Array of some shape ``n``.
        std : float or scalar Theano variable
            Standard deviation of the Gaussian noise.
        rng : Theano random number generator, optional [default: None]
            Generator to draw random numbers from. If None, rng will be
            instantiated on the spot.
        Returns
        -------
        res : Theano variable
            Of shape ``n``.
        Examples
        --------
        >>> m = T.matrix()
        >>> c = gaussian_perturb(m, 0.1)
        """


     #   noise = self.theano_rng.normal(size=arr.shape, std=std)
     #   noise = T.cast(noise, th.config.floatX)

        noise =np.random.normal(0, std, arr.shape)
        arr +=noise
        return arr


















