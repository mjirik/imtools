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


    @attr('interactive')
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

    @attr('interactive')
    def test_add_data_and_show(self):
        """
        creates VTK file from input data
        :return:
        """
        import numpy as np
        segmentation = np.zeros([10, 10, 10])
        segmentation[3:7, 2:7, 2:8] = 1
        segmentation[8, 2, 5:8] = 0
        segmentation[3:4, 4:7, 2:4] = 0
        segmentation[6:7, 2, :] = 0

        import imtools.show_segmentation_qt as ssqt
        app = QApplication(sys.argv)
        sw = ssqt.ShowSegmentationWidget(None, show_load_button=True)
        sw.add_data(segmentation)
        QTest.mouseClick(sw.ui_buttons['Show'], Qt.LeftButton)
        sw.show()
        # sw.add_vtk_file("~/projects/imtools/mesh.vtk")
        # app.exec_()


if __name__ == '__main__':
    unittest.main()
