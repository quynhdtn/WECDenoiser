__author__ = 'quynhdo'

import numpy as np
import re

class WEDict:

    def __init__(self, full_dict_path=None, full_dict=None):
        if full_dict_path is not None:
            f = open(full_dict_path, "r")
            self.full_dict = {}
            self.we_size = -1
            for l in f.readlines(): # read the full dictionary
                l = l.strip()
                tmps = re.split('\s+', l)
                if len(tmps) > 1:
                    we = []
                    if self.we_size == -1:
                        self.we_size = len(tmps)-1
                    for i in range(1, len(tmps)):
                        we.append(float(tmps[i].strip()))

                    self.full_dict[tmps[0]]= np.asarray(we, dtype="float")

            f.close()
        if full_dict is not None:
            self.full_dict = full_dict

            self.we_size = 0

    def getFullVobWE(self):
        return np.asarray([v for v in self.full_dict.values()])

    def getFullVobWEAndKeys(self):
        k = []
        t = []
        for item in self.full_dict.items():
            k.append(item[0])
            t.append(item[1])

        return np.asarray(k), np.asarray(t)

    def getWE(self, w):
        we = None
        if w in self.full_dict.keys():
            we = self.full_dict[w]
        else:
            we = np.zeros(self.we_size)
        return we

    def extractWEForVob(self, vob, output):
        f = open(output, "w")
        c = 0
        for w in vob:
            if w in self.full_dict.keys():
                f.write(w)
                f.write(" ")
                we = self.full_dict[w]
                c += 1
                for val in we:
                    f.write(str(val))
                    f.write(" ")
                f.write("\n")
        f.close()
        print ( "Words in WE dict: ")
        print (str(c) + "/" + str(len(vob)))

    def writeToFile(self, output):
        f = open(output, "w")
        for w in self.full_dict.keys():
            f.write(w)
            f.write(" ")
            for v in self.full_dict[w]:
                f.write(str(v))
                f.write(" ")
            f.write("\n")
        f.close()