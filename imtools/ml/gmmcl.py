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
            # cli = sklearn.mixture.GMM(**self.clspars)
            cli = sklearn.mixture.GaussianMixture(**self.clspars)
            dtl = data[target.reshape(-1)==label]
            cli.fit(dtl)
            self.cls[label] = cli

        pass

    def __relabel(self, target, new_keys):
        out = np.zeros(target.shape, dtype=target.dtype)
        for label, i in new_keys.iteritems():
            out[target==i] = label

        return out

    # sklearn.mixture.GaussianMixture.sc
    def scores(self, x):
        x = np.asarray(x)
        score = {}

        for label in self.cls.keys():
            # sc = self.cls[label].score(x)
            sc = self.cls[label].score_samples(x)
            score[label] = sc
        return score

    def predict(self, x):
        x = np.asarray(x)
        score_l = {}
        score = []

        for i, label in enumerate(self.cls.keys()):
            score_l[label] = i
            from distutils.version import StrictVersion

            if StrictVersion(sklearn.__version__) >= StrictVersion("0.18.0"):
                sc = self.cls[label].score_samples(x)
            else:
                logger.warning("Support for sklearn version < 0.18.0 will be removed in the future")
                sc = self.cls[label].score(x)
            score.append(sc)
        target_tmp = np.argmax(score, 0)
        return self.__relabel(target_tmp, score_l)