import pandas as pd
import numpy as np
from scipy.spatial import distance
from operator import itemgetter
import time
from sklearn.neighbors import KDTree

class DBScan():

    def __init__(self, eps, threshold, m='euclidean'):
        self.data = None        # database
        self.KDT = None
        self.e = eps            # radius
        self.t = threshold
        self.UNDEFINED = 0
        self.NOISE = 1       # minPoints: density treshold

        # cluster of i-th training example
    
    def region_query(self, p):
        N = self.KDT.query_radius([p], r=self.e)[0]  
        N = [n for n in N if self.label[n] in [self.UNDEFINED, self.NOISE] ]
        return np.array(N, dtype=object)

    def execute(self, data):
        self.data = data       # database
        self.KDT = KDTree(data) 
        self.label = np.array([self.UNDEFINED for _ in range(len(data))])        

        c = 1
        for i in range(len(self.data)):
            label_p = self.label[i]         # label of i-th training example
            if label_p != self.UNDEFINED:
                continue
            # get index of nearest neighbors of i-th trainig example
            NN = self.region_query(self.data.iloc[i])
            
            if len(NN) < self.t:
                self.label[i] = self.NOISE
                continue
            
            c += 1
            self.label[i] = c
            
            # S <- N \ {p}
            S = [NN[_] for _ in np.where(NN!=i)[0]]

            for q in S:
                label_curr = self.label[q]
                if label_curr == self.NOISE:
                    self.label[q] = c
                if label_curr != self.UNDEFINED:
                    continue
                NE = self.region_query(self.data.iloc[q])
                self.label[q] = c
                if len(NE) < self.t:
                    continue
                S += [_ for _ in NE if _ not in S]
        return self.label