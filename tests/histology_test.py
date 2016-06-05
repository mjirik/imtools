#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
import unittest
from nose.plugins.attrib import attr
import numpy as np

import logging
logger = logging.getLogger(__name__)

from imtools.tree_processing import TreeGenerator
import imtools.surface_measurement as sm


def join_sdp(datadir):

    return os.path.join(path_to_script, '../sample_data', datadir)

class HistologyTest(unittest.TestCase):
    interactiveTests = False

    @attr("LAR")
    def test_vessel_tree_lar(self):
        import imtools.gt_lar
        tvg = TreeGenerator(imtools.gt_lar.GTLar)
        yaml_path = os.path.join(path_to_script, "./hist_stats_test.yaml")
        tvg.importFromYaml(yaml_path)
        tvg.voxelsize_mm = [1, 1, 1]
        tvg.shape = [100, 100, 100]
        output = tvg.generateTree() # noqa
        if self.interactiveTests:
            tvg.show()


    def test_import_new_vt_format(self):

        tvg = TreeGenerator()
        yaml_path = os.path.join(path_to_script, "vt_biodur.yaml")
        tvg.importFromYaml(yaml_path)
        tvg.voxelsize_mm = [1, 1, 1]
        tvg.shape = [150, 150, 150]
        data3d = tvg.generateTree()

    def test_test_export_to_esofspy(self):
        """
        tests export function
        """

        import imtools.vesseltree_export as vt
        yaml_input = os.path.join(path_to_script, "vt_biodur.yaml")
        yaml_output = os.path.join(path_to_script, "delme_esofspy.txt")
        vt.vt2esofspy(yaml_input, yaml_output)


    @attr("actual")
    def test_surface_density_gensei_data(self):
        import io3d
        dr = io3d.datareader.DataReader()
        datap = dr.Get3DData(join_sdp('gensei_slices/'),
                             dataplus_format=True)
        # total object volume fraction:           0.081000
        # total object volume [(mm)^3]:           81.000000
        # total object surface fraction [1/(mm)]: 0.306450
        # total object surface [(mm)^2]:          306.449981
        segmentation = (datap['data3d'] > 100).astype(np.int8)
        voxelsize_mm = [0.2, 0.2, 0.2]
        volume = np.sum(segmentation) * np.prod(voxelsize_mm)

        Sv = sm.surface_density(segmentation, voxelsize_mm)
        self.assertGreater(volume, 80)
        self.assertLess(volume, 85)
        self.assertGreater(Sv, 0.3)
        self.assertLess(Sv, 0.4)

    def test_surface_measurement(self):

# box
        data1 = np.zeros([30, 30, 30])
        voxelsize_mm = [1, 1, 1]
        data1[10:20, 10:20, 10:20] = 1

        Sv1 = sm.surface_density(data1, voxelsize_mm)

# box without small box on corner
        data2 = np.zeros([30, 30, 30])
        voxelsize_mm = [1, 1, 1]
        data2[10:20, 10:20, 10:20] = 1
        data2[10:15, 10:15, 10:15] = 0
        Sv2 = sm.surface_density(data2, voxelsize_mm)

        self.assertEqual(Sv2, Sv1)

# box with hole in one edge
        data3 = np.zeros([30, 30, 30])
        voxelsize_mm = [1, 1, 1]
        data3[10:20, 10:20, 10:20] = 1
        data3[13:18, 13:18, 10:15] = 0
        Sv3 = sm.surface_density(data3, voxelsize_mm)
        self.assertGreater(Sv3, Sv1)
        # import sed3
        # ed = sed3.sed3(im_edg)
        # ed.show()

    def test_surface_measurement_voxelsize_mm(self):
        import scipy

# data 1
        data1 = np.zeros([30, 40, 55])
        voxelsize_mm1 = [1, 1, 1]
        data1[10:20, 10:20, 10:20] = 1
        data1[13:18, 13:18, 10:15] = 0
# data 2
        voxelsize_mm2 = [0.1, 0.2, 0.3]
        data2 = scipy.ndimage.interpolation.zoom(
            data1,
            zoom=1.0/np.asarray(voxelsize_mm2),
            order=0
        )
        # import sed3
        # ed = sed3.sed3(data1)
        # ed.show()
        # ed = sed3.sed3(data2)
        # ed.show()

        Sv1 = sm.surface_density(data1, voxelsize_mm1)
        Sv2 = sm.surface_density(data2, voxelsize_mm2)
        self.assertGreater(Sv1, Sv2*0.9)
        self.assertLess(Sv1, Sv2*1.1)

    def test_surface_measurement_use_aoi(self):
        """
        Test of AOI. In Sv2 is AOI half in compare with Sv1.
        Sv1 should be half of Sv2
        """
        data1 = np.zeros([30, 60, 60])
        aoi = np.zeros([30, 60, 60])
        aoi[:30, :60, :30] = 1
        voxelsize_mm = [1, 1, 1]
        data1[10:20, 10:20, 10:20] = 1
        data1[13:18, 13:18, 10:15] = 0

        Sv1 = sm.surface_density(data1, voxelsize_mm, aoi=None)
        Sv2 = sm.surface_density(data1, voxelsize_mm, aoi=aoi)
        self.assertGreater(2*Sv1, Sv2*0.9)
        self.assertLess(2*Sv1, Sv2*1.1)

    def test_surface_measurement_find_edge(self):
        tvg = TreeGenerator()
        yaml_path = os.path.join(path_to_script, "./hist_stats_test.yaml")
        tvg.importFromYaml(yaml_path)
        tvg.voxelsize_mm = [1, 1, 1]
        tvg.shape = [100, 100, 100]
        data3d = tvg.generateTree()

        # init histology Analyser
        # metadata = {'voxelsize_mm': tvg.voxelsize_mm}
        # data3d = data3d * 10
        # threshold = 2.5

        im_edg = sm.find_edge(data3d, 0)
        # in this area should be positive edge
        self.assertGreater(
            np.sum(im_edg[25:30, 25:30, 30]),
            3
        )
        # self.assert(im_edg
        # import sed3
        # ed = sed3.sed3(im_edg)
        # ed.show()


if __name__ == "__main__":
    unittest.main()
