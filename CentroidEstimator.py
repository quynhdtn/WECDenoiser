import theano.tensor as T
import theano as th
import numpy as np
import timeit
import os
from sklearn import preprocessing


class CentroidEstimator:

    def __init__(self, X, C, role, batch_size, training_epochs, learning_rate, output=None):
        #### C -  cluster labels of the input

        self.classes = list( np.unique(np.asarray(C)) ) # class labels
        self.num_classes = len(self.classes)
        self.dim = X.get_value().shape[1]


        Y= np.mean(X.get_value()[C[:]== role], axis=0)



        self.y = th.shared(value=np.asarray(Y, dtype=th.config.floatX),   name='y', borrow=True)
        if len(X.get_value()[C[:]== role]) == 1:
            if output is not None:
                f = open(output,  'w')

                yy = self.y.eval()
                f.write(str(role)+" ")
                for v in yy:
                    f.write(str(v) + " ")
                f.write("\n")
                f.close()
                return
            else:
                print (self.y.eval())
                return

        self.YY = T.matrix('yy')
        self.x = T.matrix(name=  "x", dtype=th.config.floatX)
        self.c = T.vector(name='c', dtype='int32')

        self.params=[self.y]
        index = T.lscalar()


        lb = preprocessing.LabelBinarizer()
        lb.fit(C)
        classes = list(lb.classes_)

        idx = classes.index(role)

        C1= lb.transform(C)

        C1= C1[:,idx]
        C1 =T.cast( np.asarray(C1), 'int32')

        # models for training
        cost,updates=self.get_cost_updates(learning_rate)

        train_model = th.function(
        [index],
        cost,
        updates=updates,
        givens={
            self.x: X[index * batch_size: (index + 1) * batch_size],
            self.c: C1[index * batch_size: (index + 1) * batch_size]

            }
        )

        # models for validating
        validate_model=None
        n_train_batches = (int) (X.get_value(borrow=True).shape[0] / batch_size)
        n_valid_batches = -1

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



                if patience <= iter:
                    done_looping = True
                    break
            print ('Training epoch %d, cost ' % epoch, np.mean(c))
        end_time = timeit.default_timer()

        print ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        if output is not None:
            f = open(output,  'w')

            yy = self.y.eval()
            f.write(str(role)+" ")
            for v in yy:
                f.write(str(v) + " ")
            f.write("\n")
            f.close()

        else:
            print (self.y.eval())

    def get_cost_updates(self, learning_rate):

        cost = 0

        k = (T.sqrt(T.sum((self.x - self.y ) **2, axis=1)))
        k1=self.c.T * k
        k2=(1-self.c.T) * k

     #   n1= k1.nonzero()[0].shape[0]
    #    n2= k2.nonzero()[0].shape[0]

        dIn= T.sum(k1)
        dOut = T.sum(k2)

        p = 0.000000001
        cost = (dIn+p)/(T.sum(self.c)+p) - (dOut+p)/(T.sum(1-self.c)+p)

        gparams = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return cost, updates

def estimateAllCentroids(X, Y, output, learning_rate=0.0001, training_epochs=100, batch_size=30):
    clusters = list( np.unique(np.asarray(Y)) ) # cluster labels

    fo = open(output, "w")
    rs={}
    for c in clusters:
        print (" Estimate for ",c)
        rf = CentroidEstimator(th.shared(X), Y, c,learning_rate=0.0001, training_epochs=10, batch_size=30, output="centroid.tmp.txt")
        f = open ("centroid.tmp.txt", "r")
        for l in f.readlines():
            fo.write(l)
            fo.write("\n")
            l=l.strip()
            tmps = l.split(" ")
            v = []
            if len(tmps)>2:
                for j in range(1, len(tmps)):
                    v.append(float(tmps[j]))
                rs[tmps[0]]=np.asarray(v)

        f.close()
        os.remove("centroid.tmp.txt")

    fo.close()

    return rs