#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Module is used for visualization of segmentation stored in pkl file.
"""

import logging
logger = logging.getLogger(__name__)
import scipy
import numpy as np
from . import ml
from . import image_manipulation as imma


def externfv(data3d, voxelsize_mm):        # scale
    f0 = scipy.ndimage.filters.gaussian_filter(data3d, sigma=3).reshape(-1, 1)
    f1 = scipy.ndimage.filters.gaussian_filter(data3d, sigma=1).reshape(-1, 1) - f0
    fv = np.concatenate([
        f0, f1
    ], 1)
    return fv

class Trainer3D():
    def __init__(self, feature_function=None):
        self.working_voxelsize_mm = [1.5, 1.5, 1.5]
        self.data=None
        self.target=None
        #         self.cl = sklearn.naive_bayes.GaussianNB()
        #         self.cl = sklearn.mixture.GMM()
        #self.cl = sklearn.tree.DecisionTreeClassifier()
        if feature_function is None:
            feature_function = externfv
        self.feature_function = feature_function
        self.cl = ml.gmmcl.GMMCl(n_components=6)

    def save(self, filename='saved.ol.p'):
        """
        Save model to pickle file
        """
        import dill as pickle
        sv = {
            # 'feature_function': self.feature_function,
            'cl': self.cl

        }
        pickle.dump(sv, open(filename, "wb"))

    def load(self, mdl_file='saved.ol.p'):
        import dill as pickle
        sv = pickle.load(open(mdl_file, "rb"))
        self.cl= sv['cl']
        # self.feature_function = sv['feature_function']


    def _fv(self, data3dr):
        return self.feature_function(data3dr, self.working_voxelsize_mm)


    def _add_to_training_data(self, data3dr, segmentationr, nth=50):
        fv = self._fv(data3dr)
        data = fv[::nth]
        target = np.reshape(segmentationr, [-1, 1])[::nth]
        #         print "shape ", data.shape, "  ", target.shape

        if self.data is None:
            self.data = data
            self.target = target
        else:
            self.data = np.concatenate([self.data, data], 0)
            self.target = np.concatenate([self.target, target], 0)
            # self.cl.fit(data, target)

            #f1[segmentationr == 0]
    def fit(self):
        #         print "sf fit data shape ", self.data.shape
        self.cl.fit(self.data, self.target)

    def predict(self, data3d, voxelsize_mm):
        data3dr = imma.resize_to_mm(data3d, voxelsize_mm, self.working_voxelsize_mm)
        fv = self._fv(data3dr)
        pred = self.cl.predict(fv)
        return imma.resize_to_shape(pred.reshape(data3dr.shape), data3d.shape)

    def predict_w(self, data3d, voxelsize_mm, weight, label0=0, label1=1):
        """
        segmentation with weight factor
        :param data3d:
        :param voxelsize_mm:
        :param weight:
        :return:
        """
        scores = self.scores(data3d, voxelsize_mm)
        out = scores[label1] > (weight * scores[label0])

        return out

    def scores(self, data3d, voxelsize_mm):
        data3dr = imma.resize_to_mm(data3d, voxelsize_mm, self.working_voxelsize_mm)
        fv = self._fv(data3dr)
        scoreslin = self.cl.scores(fv)
        scores = {}
        for key in scoreslin:
            scores[key] = imma.resize_to_shape(scoreslin[key].reshape(data3dr.shape), data3d.shape)

        return scores


    def __preprocessing(data3d):
        pass

    def add_train_data(self, data3d, segmentation, voxelsize_mm, nth=50):
        data3dr = imma.resize_to_mm(data3d, voxelsize_mm, self.working_voxelsize_mm)
        segmentationr = imma.resize_to_shape(segmentation, data3dr.shape)

        # print np.unique(segmentationr), data3dr.shape, segmentationr.shape
        self._add_to_training_data(data3dr, segmentationr, nth)
        #f1 scipy.ndimage.filters.gaussian_filter(data3dr, sigma=5)

