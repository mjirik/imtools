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

import io3d
import sed3
import imtools.trainer3d
import imtools.datasets
import sklearn
import sklearn.metrics
import sklearn.neural_network
from loguru import logger

sys.path.append(op.expanduser("~/projects/bodynavigation"))

# import bodynavigation
pt = op.expanduser("~/projects/imtools")
sys.path.append(pt)
import imtools

#

class Train3DTest(unittest.TestCase):
    interactivetTest = False
    # interactivetTest = True

    # @pytest.mark.interactive
    def test_intensity_training_ircad(self):
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

_gmm__mix_clf = imtools.ml.gmmcl.GMMCl()
_gmm__mix_clf.cl = {0:sklearn.mixture.GaussianMixture(n_components=1), 1:sklearn.mixture.GaussianMixture(n_components=3)}
@pytest.mark.parametrize('cl', [
    (sklearn.tree.DecisionTreeClassifier()),
    (_gmm__mix_clf),
    (imtools.ml.gmmcl.GMMCl()),
    (sklearn.neural_network.MLPClassifier())
])
def test_intensity_training_artificial(cl):
    slice3 = (slice(13, 27), slice(13, 17), slice(13, 27))
    shape = [30,31,32]
    voxelsize_mm = [1, 2, 3]
    d3d, seg = _mk_data(slice3, shape=shape)

    ol = imtools.trainer3d.Trainer3D(classifier=cl)
    # ol.cl = tree.DecisionTreeClassifier()
    # ol.cl = cl
    ol.add_train_data(d3d, seg, voxelsize_mm=voxelsize_mm)

    ol.fit()

    # test
    slice3 = (slice(2, 6), slice(2, 8), slice(2, 7))
    shape = [12, 11, 10]
    voxelsize_mm = [1, 2, 3]
    d3d, seg = _mk_data(slice3, shape=shape)

    pred_seg = ol.predict(d3d, voxelsize_mm)
    sed3.show_slices(d3d, contour=seg, slice_number=12)
    # ed = sed3.sed3(d3d, contour=seg)
    # ed.show()

    sc = sklearn.metrics.accuracy_score(seg.flatten(), pred_seg.flatten())
    f1 = sklearn.metrics.f1_score(seg.flatten(), pred_seg.flatten())
    logger.debug(f"f1={f1}, cl={str(cl)}")

    assert sc > 0.5
    assert f1 > 0.5





if __name__ == "__main__":
    unittest.main()
