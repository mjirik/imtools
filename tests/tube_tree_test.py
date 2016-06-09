#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import os
import os.path

from nose.plugins.attrib import attr
path_to_script = os.path.dirname(os.path.abspath(__file__))
import unittest
import numpy as np
import sys

# from imtools import qmisc
# from imtools import misc
from imtools.tree_processing import TreeGenerator
import io3d

#

class TubeTreeTest(unittest.TestCase):
    interactivetTest = False
    # interactivetTest = True

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

    def test_vessel_tree_vtk(self):
        tvg = TreeGenerator('vtk')
        yaml_path = os.path.join(path_to_script, "./hist_stats_test.yaml")
        tvg.importFromYaml(yaml_path)
        tvg.voxelsize_mm = [1, 1, 1]
        tvg.shape = [100, 100, 100]
        output = tvg.generateTree() # noqa

    @unittest.skipIf(not ("skelet3d" in sys.modules), "skelet3d is not installed")
    def test_vessel_tree_vtk_from_skeleton(self):

        import skelet3d
        import imtools
        import skelet3d.skeleton_analyser
        import shutil

        fn_out = 'tree.vkt'
        if os.path.exists(fn_out):
            shutil.rmtree(fn_out)

        volume_data = np.zeros([3, 7, 9], dtype=np.int)
        volume_data [:, :, 1:3] = 1
        volume_data [:, 5, 2:9] = 1
        volume_data [:, 0:7, 5] = 1
        skelet = skelet3d.skelet3d(volume_data)

        skan = skelet3d.skeleton_analyser.SkeletonAnalyser(skelet, volume_data=volume_data, voxelsize_mm=[1,1,1])
        stats = skan.skeleton_analysis()

        tvg = TreeGenerator('vtk')
        tvg.voxelsize_mm = [1, 1, 1]
        tvg.shape = [100, 100, 100]
        tvg.tree_data = stats
        output = tvg.generateTree() # noqa
        tvg.saveToFile(fn_out)
        os.path.exists(fn_out)


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
if __name__ == "__main__":
    unittest.main()
