#! /usr/bin/python
# -*- coding: utf-8 -*-

# import sys
import os
import sys
import os.path
import numpy as np

import logging
logger = logging.getLogger(__name__)
import sklearn
import sklearn.mixture

# path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "./extern/sPickle"))

class GMMCl():
    def __init__(self, **pars):
        self.clspars = pars
        self.cls = {}

    def fit(self, data, target):
        data = np.asarray(data)
        target = np.asarray(target).astype(np.int16)
#         print "gmmcl fit sh ", data.shape
        un = np.unique(target)
        for label in un:
            cli = sklearn.mixture.GMM(**self.clspars)
#             print "fr ", data.shape, target.shape
            dtl = data[target.reshape(-1)==label]
#             print "data target label ", data.shape, dtl.shape
            cli.fit(dtl)
            self.cls[label] = cli
            #if label in self.cls.keys():
            #    pass
            #else:

        pass

    def __relabel(self, target, new_keys):
        out = np.zeros(target.shape, dtype=target.dtype)
        for label, i in new_keys.iteritems():
            out[target==i] = label

        return out

    def scores(self, x):
        x = np.asarray(x)
#         print 'gmmcl predict shape ', x.shape
        score = {}
        #score = []

        for label in self.cls.keys():
#             print "for ", i, label, x.shape, self.cls.keys()

            sc = self.cls[label].score(x)
            score[label] = sc
        return score

    def predict(self, x):
        x = np.asarray(x)
#         print 'gmmcl predict shape ', x.shape
        score_l = {}
        score = []

        for i, label in enumerate(self.cls.keys()):
#             print "for ", i, label, x.shape, self.cls.keys()

            score_l[label] = i
            sc = self.cls[label].score(x)
            score.append(sc)
        target_tmp = np.argmax(score, 0)
        return self.__relabel(target_tmp, score_l)