#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import unittest

import pytest

import imtools
import imtools.sample_data
import imtools.show_segmentation as ss


class ShowSegmemtationCase(unittest.TestCase):
    # @pytest.mark.interactive
    def test_donut_in_one_function(self):
        datap = imtools.sample_data.donut()

        segmentation = datap['segmentation']
        voxelsize_mm = datap['voxelsize_mm']

        # app.setGraphicsSystem("openvg")
        sw = ss.showSegmentation(
            segmentation,
            degrad=1,
            # degrad=self.degrad,
            voxelsize_mm=voxelsize_mm,
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
        expected_output_fn1 = "donut_1.vtk"
        expected_output_fn2 = "donut_2.vtk"
        if op.exists(expected_output_fn1):
            os.remove(expected_output_fn1)
        if op.exists(expected_output_fn2):
            os.remove(expected_output_fn2)
        datap = imtools.sample_data.donut()

        segmentation = datap['segmentation']
        voxelsize_mm = datap['voxelsize_mm']

        # app.setGraphicsSystem("openvg")

        svtk = ss.SegmentationToMesh(segmentation)
        svtk.set_resize_parameters(degrad=1)
        svtk.set_output(filename="donut_{}.vtk", pvsm_file="donut.pvms", one_file_per_label=True)
        svtk.set_labels()
        vtk_files = svtk.make_mesh_files(
        # vtk_files = ss.prepare_vtk_files(
            # vtk_files = ss.prepare_vtk_files(
            # degrad=self.degrad,
            # voxelsize_mm=self.voxelsize_mm,
            # vtk_file="donut_{}.vtk",
            # resize_mm=self.resize_mm,
            # resize_voxel_number=self.resize_voxel_number,
            # smoothing=self.smoothing,
            # show=True
        )
        # ss.create_pvsm_file(vtk_files, "donut.pvsm")

        self.assertTrue(op.exists(expected_output_fn1))
        self.assertTrue(op.exists(expected_output_fn2))
        # self.assertTrue(op.exists("donut_1-2.pvsm"))
        # self.assertEqual(True, False)

    def test_donut_stl(self):
        expected_output_fn1 = "donut_1.stl.vtk"
        expected_output_fn2 = "donut_2.stl.vtk"
        expected_output_fn3 = "donut_1.stl"
        expected_output_fn4 = "donut_2.stl"
        if op.exists(expected_output_fn1):
            os.remove(expected_output_fn1)
        if op.exists(expected_output_fn2):
            os.remove(expected_output_fn2)
        if op.exists(expected_output_fn3):
            os.remove(expected_output_fn3)
        if op.exists(expected_output_fn4):
            os.remove(expected_output_fn4)
        datap = imtools.sample_data.donut()

        segmentation = datap['segmentation']
        voxelsize_mm = datap['voxelsize_mm']

        # app.setGraphicsSystem("openvg")

        svtk = ss.SegmentationToMesh(segmentation)
        svtk.set_resize_parameters(degrad=1)
        svtk.set_output(filename="donut_{}.stl", pvsm_file="donut.pvms", one_file_per_label=True)
        svtk.set_labels()
        vtk_files = svtk.make_mesh(
        )

        self.assertTrue(op.exists(expected_output_fn1))
        self.assertTrue(op.exists(expected_output_fn2))
        self.assertTrue(op.exists(expected_output_fn3))
        self.assertTrue(op.exists(expected_output_fn4))
        # self.assertTrue(op.exists("donut_1-2.pvsm"))
        # self.assertEqual(True, False)
        os.remove(expected_output_fn1)
        os.remove(expected_output_fn2)
        os.remove(expected_output_fn3)
        os.remove(expected_output_fn4)

    @pytest.mark.slow
    # @pytest.mark.interactive
    def test_from_file(self):
        input_file = "~/lisa_data/jatra_5mm_new.pklz"
        import io3d.datasets
        input_file = io3d.datasets.join_path(r"3Dircadb1.1/MASKS_DICOM/liver")
        output_file = "jatra.vtk"
        if op.exists(output_file):
            os.remove(output_file)

        input_file = op.expanduser(input_file)

        import io3d
        datap = io3d.datareader.read(input_file, dataplus_format=True)


        segmentation = datap["data3d"]
        voxelsize_mm = datap['voxelsize_mm']

        # app.setGraphicsSystem("openvg")
        sw = ss.showSegmentation(
            # (segmentation==1).astype(np.int8),
            segmentation=(segmentation > 0),
            degrad=1,
            labels=[1],
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
        # clean
        os.remove(output_file)

    def test_create_pmvs(self):
        vtk_files = ["file1.vtk", "file2.vtk"]
        pvsm_file = "test_delete.pvsm"
        imtools.show_segmentation.create_pvsm_file(vtk_files, pvsm_file)
        self.assertTrue(op.exists(pvsm_file))
        os.remove(pvsm_file)

if __name__ == '__main__':
    unittest.main()
