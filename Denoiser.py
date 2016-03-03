import argparse
from WEDict import WEDict
from nn.Functions import CrossEntropyCostFunction
from nn.Layer import Layer
from nn.NNNet import NNNet
from CentroidEstimator import estimateAllCentroids

__author__ = 'quynhdo'

import numpy as np
import theano as th


__author__ = 'quynhdo'
class WEDenoiser(NNNet):
   '''
   this DA is used for SRL roles where the orginal WE of roles are used as the corrupted version and the average vector of all the roles are used as good value
   '''

   def __init__(self, nIn=700, nHidden=500,  idx="", initial_w=None, initial_b=None, x=None,
                y=None, cost_function=CrossEntropyCostFunction, we_file=None, we_dict=None):
        self.idx = idx
        self.nIn = nIn
        self.nHidden = nHidden
        NNNet.__init__(self, costfunction=cost_function, idx=idx,x=x, y=y)

        ilayer = Layer(numNodes=nIn, ltype = Layer.Layer_Type_Input)
        hlayer = Layer(numNodes=nHidden, ltype = Layer.Layer_Type_Hidden)
        olayer = Layer(numNodes=nIn, ltype = Layer.Layer_Type_Output)
        self.addLayer([ilayer, hlayer, olayer])

        # encoding connection
        if initial_w is None:
            rng = np.random.RandomState(123)
            w_lamda = 40 * np.sqrt(6. / (nIn + nHidden))
       #     w_lamda = 0.1
            encode_conn = self.createConnection(ilayer, hlayer, w_lamda=w_lamda, rng=rng, initial_b=initial_b) # the initial bias if exis is only for encoding phase
        else:
            encode_conn = self.createConnection(ilayer, hlayer, initial_w=initial_w, initial_b=initial_b)
        decode_conn = self.createConnection(hlayer, olayer)
        self.params.remove(decode_conn.W)
        decode_conn.W = encode_conn.W.T
        self.weDict = None
        if we_file is not None:
            self.weDict = WEDict(full_dict_path = we_file)
        else:
            if we_dict is not None:
                self.weDict = we_dict
        self.centroids={}
        self.start()


   def fit(self, train_data, train_data_label,  batch_size, training_epochs, learning_rate, save_model_path= None,regularization_name=None, regularization_lambda=0.0001):
       '''

       :param train_data: a list of words
       :param train_data_label:
       :param batch_size:
       :param training_epochs:
       :param learning_rate:
       :param save_model_path:
       :param regularization_name:
       :param regularization_lambda:
       :return:
       '''
       Xi = np.asarray([self.weDict.getWE(xd) for xd in train_data])
       self.centroids = estimateAllCentroids(Xi,np.asarray(train_data_label), self.idx+"centroids.txt")

       role_list =  np.unique(train_data_label)
       for r in role_list:
           if r not in self.centroids.keys():
               train_x = [train_data[i] for i in range(len(train_data)) if train_data_label[i]==r]
               avgvector = np.mean(train_x, axis=0)
               self.centroids[r] = avgvector
       y=[]
       for i in range(len(train_data)):
           y.append(self.centroids[train_data_label[i]])

       X = th.shared(Xi, borrow=True)
       Y=th.shared(np.asarray(y, dtype=th.config.floatX), borrow=True)
       NNNet.fit(self, train_data=X, train_data_label=Y, training_epochs=training_epochs,
                 learning_rate=learning_rate, save_model_path=save_model_path, batch_size=batch_size,
                 regularization_name=regularization_name,
                 regularization_lambda=regularization_lambda)

   def exportNewWE(self):
       newWeDict= {}
       k,wes = self.weDict.getFullVobWEAndKeys()
       nwes = self.getHidden(wes).eval()
       for i in range(len(k)):
           newWeDict[k[i]]= np.asarray(nwes[i])

       return WEDict(full_dict=newWeDict)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("nIn", help="Input size")
    parser.add_argument("nHidden", help="Hidden size")
    parser.add_argument("weFile", help="Train data file")
    parser.add_argument("trainFile", help="Train data file")
    parser.add_argument("outputFile", help="Train data file")
    parser.add_argument("-b", "--b", type=int,
                    help="batch size")
    parser.add_argument("-n", "--n", type=int,
                    help="num epoches")
    parser.add_argument("-l", "--l", type=float,
                    help="learning rate")

    parser.add_argument("-r", "--r", type=str,
                    help="regularization name", default=None)

    parser.add_argument("-rl", "--rl", type=float,
                    help="regularization lambda", default=0.0001)
    args = parser.parse_args()

    f = open(args.trainFile, "r")

    X=[]
    Y=[]
    for line in f.readlines():
        line = line.strip()
        tmps = line.split(" ")
        if len(tmps) > 0:
            X.append(tmps[0])
            Y.append(tmps[len(tmps)-1])
    f.close()



    wd = WEDenoiser(nIn=int(args.nIn),nHidden= int(args.nHidden), we_file=args.weFile)


    wd.fit(X, Y, args.b, args.n, args.l, None,args.r, args.rl)
    newwe = wd.exportNewWE()
    newwe.writeToFile(args.outputFile)


