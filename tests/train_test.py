#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import unittest
import numpy as np
import os
import os.path
import pytest
import os.path as op
import sys
import matplotlib.pyplot as plt
import glob
import itertools

import io3d
import sed3
import imtools.trainer3d
import imtools.datasets
import sklearn
import sklearn.metrics
import sklearn.neural_network
from sklearn.svm import SVC
from loguru import logger

sys.path.append(op.expanduser("~/projects/bodynavigation"))

# import bodynavigation
pt = op.expanduser("~/projects/imtools")
sys.path.append(pt)
import imtools

# @pytest.mark.interactive
def test_intensity_training_ircad():
    #nth - use every nth pixel
    nth = 10
    # TODO use ircad
    sliver_reference_dir = io3d.datasets.joinp("~/data/medical/orig/sliver07/training/", return_as_str=True)
    # sliver_reference_dir = op.expanduser("~/data/medical/orig/sliver07/training/")
    # Train
    ol = imtools.trainer3d.Trainer3D()
    # ol.feature_function = localization_fv

    # for one in imtools.datasets.sliver_reader("*[0].mhd", read_seg=True):
    for i in range(1, 2):
        datap = io3d.read_dataset('3Dircadb1', "data3d", i)
        datap_liver = io3d.read_dataset('3Dircadb1', "liver", i)
        ol.add_train_data(datap["data3d"], (datap_liver["data3d"] > 0).astype(np.uint8), voxelsize_mm=datap["voxelsize_mm"])
        # numeric_label, vs_mm, oname, orig_data, rname, ref_data = one
        # ol.add_train_data(orig_data, ref_data, voxelsize_mm=vs_mm)

    ol.fit()

    # Testing
    i = 1
    datap = io3d.datasets.read_dataset("3Dircadb1", 'data3d', i)
    datap_liver = io3d.datasets.read_dataset("3Dircadb1", 'liver', i)
    data3d = datap["data3d"]
    segmentation = (datap_liver["data3d"] > 0).astype(np.uint8)
    fit = ol.predict(data3d, voxelsize_mm=datap["voxelsize_mm"])

    # one = list(imtools.datasets.sliver_reader("*018.mhd", read_seg=True))[0]
    # numeric_label, vs_mm, oname, orig_data, rname, ref_data = one
    # fit = ol.predict(orig_data, voxelsize_mm=vs_mm)

    err = segmentation != (fit > 0).astype(np.uint8)
    # visualization
    # plt.figure(figsize=(15, 10))
    # sed3.show_slices(datap["data3d"], fit, slice_step=20, axis=1, flipV=True)

    accuracy = np.sum(~err) / np.prod(data3d.shape)
    assert accuracy >= 0.80


        # assert




def _mk_data(slice3, offset=1, shape=[10, 11, 12]):
    data3d = np.random.random(shape)
    data3d[slice3] += offset

    segmentation = np.zeros(shape, dtype=int)
    segmentation[slice3] = 1
    return data3d, segmentation

# TODO finish product
_gmm__mix_clf = imtools.ml.gmmcl.GMMCl()
_gmm__mix_clf.cl = {0:sklearn.mixture.GaussianMixture(n_components=1), 1:sklearn.mixture.GaussianMixture(n_components=3)}
@pytest.mark.parametrize('cl,shape', itertools.product(
    [
        # sklearn.tree.DecisionTreeClassifier(),
        # _gmm__mix_clf,
        imtools.ml.gmmcl.GMMCl(),
        # sklearn.neural_network.MLPClassifier(),
        SVC(kernel='linear', class_weight='balanced', probability=True),
        # SVC()
    ],
    [
        # [10, 11, 12],
        [30, 31, 32],
    ]
))
def test_intensity_training_artificial(cl, shape):
    """
    Test different classifiers on unbalanced dataset.
    :param cl:
    :param shape:
    :return:
    """
    # scl = str(cl)
    # logger.debug(f'cl={scl[:min(30, len(scl))]}')
    logger.debug(f'cl={cl}')
    logger.debug(f'shape={shape}')
    slice3 = (slice(3, 7), slice(3, 7), slice(3, 7))
    # shape = [30,31,32]
    voxelsize_mm = [1, 2, 3]
    d3d, seg = _mk_data(slice3, shape=shape, offset=0.7)
    un, counts = np.unique(seg.flatten(), return_counts=True)
    logger.debug(f'counts={counts}')
    ol = imtools.trainer3d.Trainer3D(classifier=cl)
    ol.working_voxelsize_mm=[2,2,2]
    # ol.cl = tree.DecisionTreeClassifier()
    # ol.cl = cl
    ol.add_train_data(d3d, seg, voxelsize_mm=voxelsize_mm, nth=None)  # We take all samples

    #  https://elitedatascience.com/imbalanced-classes


    un, counts = np.unique(ol.target, return_counts=True)
    n_samples = np.min(counts)
    new_data_list = []
    new_target_list = []
    for label in un:
        all_data_for_one_label = ol.data[ol.target.astype(np.uint8).flatten() == label]
        # TODO mozna pouzit funkci sklearn.utils.resample
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        resamples = sklearn.utils.resample(all_data_for_one_label, n_samples=n_samples, replace=True)
        # data_subset = all_data_for_one_label[:n_samples]  # pick first n samples
        # new_data_list.append(data_subset)
        new_data_list.append(resamples)
        new_target_list.append(np.ones([n_samples], dtype=type(label)) * label)



    original_data = ol.data
    original_target = ol.target
    new_data = np.concatenate(new_data_list, axis=0)
    new_target = np.concatenate(new_target_list, axis=0)

    ol.data = new_data
    ol.target = new_target

    ol.fit()

    # test
    # slice3 = (slice(2, 6), slice(2, 8), slice(2, 7))
    # shape = [12, 11, 10]
    # voxelsize_mm = [1, 2, 3]
    d3d, seg = _mk_data(slice3, shape=shape, offset=0.7)

    pred_seg = ol.predict(d3d, voxelsize_mm)
    sed3.show_slices(d3d, contour=seg, slice_number=8)
    # ed = sed3.sed3(d3d, contour=seg)
    # ed.show()

    sc = sklearn.metrics.accuracy_score(seg.flatten(), pred_seg.flatten())
    f1 = sklearn.metrics.f1_score(seg.flatten(), pred_seg.flatten())
    logger.debug(f"f1={f1}")

    assert sc > 0.5
    assert f1 > 0.5



def test_resample():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    X = np.random.rand(y.shape[0]) + y
    X = X.reshape([-1,1])
    # X = np.array([[1., 0.], [2., 1.], [0., 0.]])
    # y = np.array([0, 1, 2])

    # from scipy.sparse import coo_matrix
    # X_sparse = coo_matrix(X)

    from sklearn.utils import shuffle, resample
    # X, X_sparse, y = shuffle(X, X_sparse, y, random_state=0)
    # X
    #
    # X_sparse
    #
    # X_sparse.toarray()

    y

    # shuffle(y, n_samples=2, random_state=0)

    Xr = resample(X, n_samples=2, random_state=0)
    print(Xr)


def balance_dataset(X,y):
    labels, counts = np.unique(y)

