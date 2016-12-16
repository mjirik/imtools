#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from nose.plugins.attrib import attr

import sys
import PyQt4
from PyQt4.QtGui import QApplication, QFileDialog
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt

class MyTestCase(unittest.TestCase):
    @attr('interactive')
    def test_something(self):
        self.assertEqual(True, False)

    @attr('interactive')
    def test_visualization(self):

        import PyQt4
        from PyQt4.QtGui import QApplication, QFileDialog
        # from teigen.dictwidgetqt import DictWidget
        # from teigen.gui import TeigenWidget
        import imtools.show_segmentation_qt
        app = QApplication(sys.argv)
        cfg = {"bool": True, "int":5, 'str': 'strdrr', 'vs':[1.0, 2.5, 7]}
        captions = {"int": "toto je int"}
        cw = imtools.show_segmentation_qt.ShowSegmentationWidget(None)
        cw.show()
        app.exec_()

    @attr('interactive')
    def test_showsegmentation_andclose(self):

        import PyQt4
        from PyQt4.QtGui import QApplication, QFileDialog
        # from teigen.dictwidgetqt import DictWidget
        # from teigen.gui import TeigenWidget
        import imtools.show_segmentation_qt
        app = QApplication(sys.argv)
        cfg = {"bool": True, "int":5, 'str': 'strdrr', 'vs':[1.0, 2.5, 7]}
        captions = {"int": "toto je int"}
        cw = imtools.show_segmentation_qt.ShowSegmentationWidget(None)

        cw.show()
        cw.close()
        # app.exec_()

    @attr('interactive')
    def test_show_segmentation_qt_widget(self):
        # from teigen.dictwidgetqt import DictWidget
        # from teigen.generators.cylindersqt import CylindersWidget
        import imtools.show_segmentation_qt as ssqt
        app = QApplication(sys.argv)
        sw = ssqt.ShowSegmentationWidget(None)
        sw.show()
        app.exec_()


    # @attr('interactive')
    def test_show_segmentation_qt_widget_hidden_buttons(self):
        # = np.zeros([10, 10, 10])
        import imtools
        import imtools.sample_data
        # imtools.sam
        # imtools.sample_data.get_sample_data("sliver_training_001")
        # from teigen.dictwidgetqt import DictWidget
        # from teigen.generators.cylindersqt import CylindersWidget
        import imtools.show_segmentation_qt as ssqt
        app = QApplication(sys.argv)
        # sw = ssqt.ShowSegmentationWidget(None, show_buttons=False)
        sw = ssqt.ShowSegmentationWidget(None, show_load_button=True)
        self.assertIn("add_data_file", sw.ui_buttons.keys())
        sw.show()
        app.exec_()

    # @attr('interactive')
    def test_add_data_and_show(self):
        """
        creates VTK file from input data
        :return:
        """
        import numpy as np
        segmentation = np.zeros([20, 30, 40])
        # generate test data
        segmentation[6:10, 7:24, 10:37] = 1
        segmentation[6:10, 7, 10] = 0
        segmentation[6:10, 23, 10] = 0
        segmentation[6:10, 7, 36] = 0
        segmentation[6:10, 23, 36] = 0
        segmentation[2:18, 12:19, 18:28] = 2

        data3d = segmentation * 100 + np.random.random(segmentation.shape) * 30
        voxelsize_mm=[3,2,1]

        import io3d
        datap = {
            'data3d': data3d,
            'segmentation': segmentation,
            'voxelsize_mm': voxelsize_mm
        }
        io3d.write(datap, "donut.pklz")

        import imtools.show_segmentation_qt as ssqt
        app = QApplication(sys.argv)
        sw = ssqt.ShowSegmentationWidget(None, show_load_button=True)
        sw.smoothing = False
        sw.add_data(segmentation, voxelsize_mm=voxelsize_mm)
        QTest.mouseClick(sw.ui_buttons['Show volume'], Qt.LeftButton)
        # sw.add_vtk_file("~/projects/imtools/mesh.vtk")
        sw.show()
        sw.close()
        # app.exec_()


if __name__ == '__main__':
    unittest.main()
