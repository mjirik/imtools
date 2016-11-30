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
import imtools.tree_processing
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


    def test_show_segmentation_qt_widget(self):
        import PyQt4
        from PyQt4.QtGui import QApplication, QFileDialog
        # from teigen.dictwidgetqt import DictWidget
        # from teigen.generators.cylindersqt import CylindersWidget
        import imtools.show_segmentation_qt as ssqt
        app = QApplication(sys.argv)
        sw = ssqt.ShowSegmentationWidget(None)
        sw.show()
        app.exec_()

    def test_show_segmentation_qt_widget_hidden_buttons(self):
        import PyQt4
        from PyQt4.QtGui import QApplication, QFileDialog
        # from teigen.dictwidgetqt import DictWidget
        # from teigen.generators.cylindersqt import CylindersWidget
        import imtools.show_segmentation_qt as ssqt
        app = QApplication(sys.argv)
        sw = ssqt.ShowSegmentationWidget(None, show_buttons=False)
        # sw.show()
        sw.show_file("~/projects/imtools/mesh.vtk")
        app.exec_()


if __name__ == "__main__":
    unittest.main()
