#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys


import unittest



import matplotlib.pyplot as plt


import numpy as np


from PyQt5.QtWidgets import QApplication
import pytest



import imtools


import imtools.sample_data


import imtools.thresholding_functions


import imtools.uiThreshold




class MyTestCase(unittest.TestCase):
    @pytest.mark.interactive
    def test_something(self):
        self.assertEqual(True, False)

    # @unittest.skipIf(os.environ.get("TRAVIS", True), "Skip on Travis-CI")
    def test_threshold(self):
        datap = imtools.sample_data.generate()
        uit = imtools.uiThreshold.uiThreshold(datap['data3d'], datap['voxelsize_mm'], interactivity=False, threshold=100)
        uit.run()

    def test_threshold_with_seed(self):
        datap = imtools.sample_data.generate()
        uit = imtools.uiThreshold.uiThreshold(datap['data3d'], datap['voxelsize_mm'], interactivity=False, seeds=datap["seeds_porta"])
        uit.run()

    def test_threshold_image_processing(self):
        datap = imtools.sample_data.generate()
        imthr, updated_threshold = imtools.uiThreshold.make_image_processing(
            data=datap['data3d'],
            voxelsize_mm=datap['voxelsize_mm'],
            # seeds=np.nonzero(datap["seeds_porta"]),
            seeds=datap["seeds_porta"],
            sigma_mm=1,
            min_threshold=None, max_threshold=None,
            closeNum=0, openNum=0,
            min_threshold_auto_method="",
            fill_holes=True, get_priority_objects=True,
            nObj=1
        )

        golden_true_porta = datap["segmentation"] == datap["slab"]["porta"]
        found_porta = imthr > 0

        err = np.sum(np.abs(golden_true_porta.astype(np.int8) - found_porta.astype(np.int8)))
        err_percent = err / np.prod(datap["data3d"].shape)
        self.assertLess(err_percent, 0.1)

    @pytest.mark.interactive
    def test_ui_threshold(self):
        datap = imtools.sample_data.generate()
        uit = imtools.uiThreshold.uiThreshold(datap['data3d'], datap['voxelsize_mm'], interactivity=True, threshold=100)
        uit.run()
        plt.show()

    # def test_gui_constructor(self):
    #     datap = imtools.sample_data.generate()
    #     uit = imtools.uiThreshold.uiThreshold(datap['data3d'], datap['voxelsize_mm'], interactivity=True, threshold=100)


    @pytest.mark.interactive
    def test_ui_threshold_qt(self):
        app = QApplication(sys.argv)
        datap = imtools.sample_data.generate()
        uit = imtools.uiThreshold.uiThresholdQt(datap['data3d'], datap['voxelsize_mm'], interactivity=True, threshold=100)

        uit.run()
        # plt.show()

    def test_getPriorityObject(self):
        import skimage.morphology


        nobj = 2
        datap = imtools.sample_data.generate()
        thresholded = datap["data3d"] > 80
        selection = imtools.thresholding_functions.get_priority_objects(thresholded, nObj=nobj, seeds_multi_index=None)

        lab = skimage.morphology.label(selection)
        output_nobj = np.unique(lab)

        self.assertEqual(nobj, nobj)

    def test_getPriorityObjectSeeds(self):
        import skimage.morphology



        nobj = 1
        datap = imtools.sample_data.generate()
        thresholded = datap["data3d"] > 80
        seeds_multi_index = np.nonzero(datap['seeds'] == 1)

        selection = imtools.thresholding_functions.getPriorityObjects(thresholded, nObj=nobj, seeds_multi_index=seeds_multi_index)

        lab = skimage.morphology.label(selection)
        output_nobj = len(np.unique(lab))

        self.assertEqual(output_nobj, nobj + 1)  # one is for background label

    @unittest.skip("not completed")
    def test_thresholding(self):
        import skimage.morphology


        datap = imtools.sample_data.generate()

        max_threshold = None
        selection = imtools.thresholding_functions.thresholding(
            datap['data3d'],
            min_threshold=80,
            max_threshold=max_threshold,
            use_min_threshold=True,
            use_max_threshold=max_threshold is None
        )
        lab = skimage.morphology.label(selection)
        output_nobj = len(np.unique(lab))

        self.assertEqual(output_nobj, 1)

    # def test_resize_to_shape_wiht_zoom_no_new_unique_values(self):
    #
    #     data = np.zeros([10, 15, 12])
    #     value1 = 1
    #     value2 = 2
    #     data[:5, :7, :6] = value1
    #     data[-5:, :7, :6] = value2
    #
    #     expected_shape = [15, 15, 15]
    #     zoom = data.shape / np.array(expected_shape)
    #     resized = imtools.thresholding_functions.resize_to_shape_with_zoom(data, expected_shape, zoom=zoom)
    #     unique = np.unique(resized)
    #
    #     self.assertEqual(resized.shape[0], expected_shape[0])
    #     self.assertEqual(resized.shape[1], expected_shape[1])
    #     self.assertEqual(resized.shape[2], expected_shape[2])
    #     self.assertEqual(resized[1, 1, 1], value1)
    #     self.assertEqual(resized[-2, 1, 1], value2)
    #     self.assertEqual(len(unique), 3)
    #     self.assertEqual(unique[0], 0)
    #     self.assertEqual(unique[1], 1)
    #     self.assertEqual(unique[2], 2)

if __name__ == '__main__':
    unittest.main()
