#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from nose.plugins.attrib import attr

import sys
import os
import os.path as op
import numpy as np

import unittest
import imtools
import imtools.sample_data
import imtools.show_segmentation as ss


class ShowSegmemtationCase(unittest.TestCase):
    # @attr('interactive')
    def test_donut(self):
        datap = imtools.sample_data.donut()

        segmentation = datap['segmentation']
        voxelsize_mm = datap['voxelsize_mm']

        import imtools.show_segmentation_qt as ssqt
        # app.setGraphicsSystem("openvg")
        sw = ss.showSegmentation(
            segmentation,
            degrad=1,
            # degrad=self.degrad,
            # voxelsize_mm=self.voxelsize_mm,
            vtk_file="donut.vtk",
            # resize_mm=self.resize_mm,
            # resize_voxel_number=self.resize_voxel_number,
            # smoothing=self.smoothing,
            show=False
            # show=True
        )

        self.assertTrue(op.exists("donut.vtk"))
        # self.assertEqual(True, False)

    def test_donut(self):
        datap = imtools.sample_data.donut()

        segmentation = datap['segmentation']
        voxelsize_mm = datap['voxelsize_mm']

        import imtools.show_segmentation_qt as ssqt
        # app.setGraphicsSystem("openvg")
        vtk_files = ss.prepare_vtk_files(
            segmentation,
            degrad=1,
            # degrad=self.degrad,
            # voxelsize_mm=self.voxelsize_mm,
            vtk_file="donut_{}.vtk",
            # resize_mm=self.resize_mm,
            # resize_voxel_number=self.resize_voxel_number,
            # smoothing=self.smoothing,
            # show=True
        )
        ss.create_pvsm_file(vtk_files, "donut.pvsm")

        self.assertTrue(op.exists("donut_1.vtk"))
        self.assertTrue(op.exists("donut_2.vtk"))
        # self.assertTrue(op.exists("donut_1-2.pvsm"))
        # self.assertEqual(True, False)

    @attr('long')
    @attr('interactive')
    def test_from_file(self):
        input_file = "~/lisa_data/jatra_5mm_new.pklz"
        output_file = "jatra.vtk"


        input_file = op.expanduser(input_file)

        import io3d
        datap = io3d.datareader.read(input_file, dataplus_format=True)


        segmentation = datap['segmentation']
        voxelsize_mm = datap['voxelsize_mm']

        import imtools.show_segmentation_qt as ssqt
        # app.setGraphicsSystem("openvg")
        sw = ss.showSegmentation(
            # (segmentation==1).astype(np.int8),
            segmentation==1,
            degrad=1,
            label=[1],
            # degrad=self.degrad,
            voxelsize_mm=voxelsize_mm,
            vtk_file=output_file,
            # resize_mm=self.resize_mm,
            resize_voxel_number=90000,
            # smoothing=self.smoothing,
            show=False
        )
        # import sed3
        #
        # ed = sed3.sed3(sw.astype(np.float))
        # ed.show()

        self.assertTrue(op.exists(output_file))

    def test_create_pmvs(self):
        vtk_files = ["file1.vtk", "file2.vtk"]
        pvsm_file = "test_delete.pvsm"
        imtools.show_segmentation.create_pvsm_file(vtk_files, pvsm_file)
        self.assertTrue(op.exists(pvsm_file))
        os.remove(pvsm_file)

if __name__ == '__main__':
    unittest.main()
