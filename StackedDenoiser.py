import argparse
from Denoiser import WEDenoiser
from WEDict import WEDict

__author__ = 'quynhdo'

class WEStackedDenoiser:

    def __init__(self, wefile, nIn, hidden_layer_sizes):

        self.denoisers= []
        nI = nIn
        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                hs = hidden_layer_sizes[i]
                d = WEDenoiser(we_file=wefile, nIn=nI, nHidden=hs)
                self.denoisers.append(d)
                nI = hs
            else:
                hs = hidden_layer_sizes[i]
                d = WEDenoiser(nIn=nI, nHidden=hs)
                self.denoisers.append(d)
                nI = hs

    def fit(self, train_data, train_data_label,  batch_size, training_epochs, learning_rate,
            save_model_path= None,regularization_name=None, regularization_lambda=0.0001):
        self.denoisers[0].fit(train_data, train_data_label,  batch_size[0], training_epochs[0], learning_rate[0],
            None,regularization_name[0], regularization_lambda[0])

        weD = self.denoisers[0].exportNewWE()

        if len(batch_size) == 1:
            batch_size = [batch_size[0]] * len(self.denoisers)

        if len(training_epochs) == 1:
            training_epochs = [training_epochs[0]] * len(self.denoisers)

        if len(learning_rate) == 1:
            learning_rate = [learning_rate[0]] * len(self.denoisers)

        if len(regularization_name) == 1:
            regularization_name = [regularization_name[0]] * len(self.denoisers)


        if len(regularization_lambda) == 1:
            regularization_lambda = [regularization_lambda[0]] * len(self.denoisers)


        for i in range(1, len(self.denoisers)):
            self.denoisers[i].weDict= weD
            print(weD.full_dict['he'])
            self.denoisers[i].fit(train_data, train_data_label,  batch_size[i], training_epochs[i], learning_rate[i],
            None,regularization_name[i], regularization_lambda[i])
            weD = self.denoisers[i].exportNewWE()


    def exportNewWE(self):
       newWeDict= {}
       k,wes = self.denoisers[0].weDict.getFullVobWEAndKeys()
       for d in self.denoisers:
           wes = d.getHidden(wes).eval()
       for i in range(len(k)):
           newWeDict[k[i]]= wes[i]

       return WEDict(full_dict=newWeDict)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("nIn", help="Input size")
    parser.add_argument("nHiddens", help="Hidden sizes")
    parser.add_argument("weFile", help="Train data file")
    parser.add_argument("trainFile", help="Train data file")
    parser.add_argument("outputFile", help="Train data file")
    parser.add_argument("-b", "--b", type=str,
                    help="batch size")
    parser.add_argument("-n", "--n", type=str,
                    help="num epoches")
    parser.add_argument("-l", "--l", type=str,
                    help="learning rate")

    parser.add_argument("-r", "--r", type=str,
                    help="regularization name", default="")

    parser.add_argument("-rl", "--rl", type=str,
                    help="regularization lambda", default="0.0001")
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
    tmps = args.nHiddens.split(",")
    args.nHiddens = [ int (x) for x in tmps]
    tmps = args.b.split(",")
    args.b = [ int (x) for x in tmps]

    tmps = args.n.split(",")
    args.n = [ int (x) for x in tmps]

    tmps = args.l.split(",")
    args.l= [ float(x) for x in tmps]

    tmps = args.rl.split(",")
    args.rl= [float(x) for x in tmps]
    if len(args.rl) == 0:
        args.rl = [0.0001]

    tmps = args.r.split(",")
    args.r =[x for x in tmps if x != ""]
    if len(args.r) == 0:
        args.r = [None]

    wd = WEStackedDenoiser(nIn=int(args.nIn), hidden_layer_sizes=args.nHiddens, wefile=args.weFile)


    wd.fit(X, Y, args.b, args.n, args.l, None,args.r, args.rl)
    newwe = wd.exportNewWE()
    newwe.writeToFile(args.outputFile)


