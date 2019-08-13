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


sys.path.append(op.expanduser("~/projects/bodynavigation"))

# import bodynavigation
pt = op.expanduser("~/projects/imtools")
sys.path.append(pt)
import imtools

#

class Train3DTest(unittest.TestCase):
    interactivetTest = False
    # interactivetTest = True

    @pytest.mark.interactive
    def test_intensity_training(self):
        # TODO use ircad
        sliver_reference_dir = op.expanduser("~/data/medical/orig/sliver07/training/")
        # Train
        import imtools.trainer3d
        import imtools.datasets
        ol = imtools.trainer3d.Trainer3D()
        # ol.feature_function = localization_fv

        for one in imtools.datasets.sliver_reader("*[0].mhd", read_seg=True):
            numeric_label, vs_mm, oname, orig_data, rname, ref_data = one
            ol.add_train_data(orig_data, ref_data, voxelsize_mm=vs_mm)

        ol.fit()

        # Testing

        one = list(imtools.datasets.sliver_reader("*018.mhd", read_seg=True))[0]
        numeric_label, vs_mm, oname, orig_data, rname, ref_data = one
        fit = ol.predict(orig_data, voxelsize_mm=vs_mm)

        # visualization
        plt.figure(figsize=(15,10))
        sed3.show_slices(orig_data, fit, slice_step=20, axis=1, flipV=True)

if __name__ == "__main__":
    unittest.main()
