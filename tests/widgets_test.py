#! /usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import os.path as op
import sys
import unittest

from PyQt5.QtCore import Qt


from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtTest import QTest


import pytest

import imtools.sample_data


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass
        # self.qapp = QApplication(sys.argv)

    @pytest.mark.interactive
    def test_something(self):
        self.assertEqual(True, False)

    @pytest.mark.interactive
    def test_visualization(self):

        from PyQt5.QtWidgets import QApplication
        # from teigen.dictwidgetqt import DictWidget
        # from teigen.gui import TeigenWidget
        import imtools.show_segmentation_qt


        app = QApplication(sys.argv)
        cfg = {"bool": True, "int":5, 'str': 'strdrr', 'vs':[1.0, 2.5, 7]}
        captions = {"int": "toto je int"}
        cw = imtools.show_segmentation_qt.ShowSegmentationWidget(None)
        cw.show()
        app.exec_()

    @pytest.mark.interactive
    def test_showsegmentation_andclose(self):

        from PyQt5.QtWidgets import QApplication
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

    @pytest.mark.interactive
    def test_show_segmentation_qt_widget(self):
        # from teigen.dictwidgetqt import DictWidget
        # from teigen.generators.cylindersqt import CylindersWidget
        import imtools.show_segmentation_qt as ssqt


        # app = QtGui.QApplication(sys.argv)
        app = QApplication(sys.argv)
        sw = ssqt.ShowSegmentationWidget(None, show_load_interface=True)
        sw.show()
        app.exec_()

    # def test_show_segmentation_qt_widget_2(self):
        # import imtools.show_segmentation_qt as ssqt
        # app = QApplication(sys.argv)
        # slab = {"la": 5, "sdfa":7}
        # sw = ssqt.SelectLabelWidget(slab=slab)
        # sw.show()
        # app.exec_()

    @pytest.mark.interactive
    def test_show_qtwidget(self):
        # from teigen.dictwidgetqt import DictWidget
        # from teigen.generators.cylindersqt import CylindersWidget
        app = QApplication(sys.argv)
        qw = QWidget()
        button = QPushButton('Test')
        layout = QVBoxLayout()
        layout.addWidget(button)
        # sw = ssqt.ShowSegmentationWidget(None, qapp=app)
        qw.show()
        app.exec_()

    # @pytest.mark.interactive
    @unittest.skipIf(os.environ.get("TRAVIS", True), "Skip on Travis-CI")
    def test_show_segmentation_qt_widget_hidden_buttons(self):
        # = np.zeros([10, 10, 10])
        # imtools.sam
        # imtools.sample_data.get_sample_data("sliver_training_001")
        # from teigen.dictwidgetqt import DictWidget
        # from teigen.generators.cylindersqt import CylindersWidget
        import imtools.show_segmentation_qt as ssqt


        app = QApplication(sys.argv)
        # app = QApplication([])

        # if "TRAVIS" in os.environ:
        #     app.setGraphicsSystem("openvg")
        # sw = ssqt.ShowSegmentationWidget(None, show_buttons=False)
        sw = ssqt.ShowSegmentationWidget(None, show_load_interface=True)
        self.assertIn("add_data_file", sw.ui_buttons.keys())
        sw.show()
        # app.exec_()
        sw.close()
        sw.deleteLater()
        sw = None
        app.quit()
        app.deleteLater()
        # app.quit()
        # app.exit()

    @pytest.mark.interactive
    def test_show_donut(self):
        """
        creates VTK file from input data
        :return:
        """
        datap = imtools.sample_data.donut()

        segmentation = datap['segmentation']
        voxelsize_mm = datap['voxelsize_mm']

        import imtools.show_segmentation_qt as ssqt


        app = QApplication(sys.argv)
        # app.setGraphicsSystem("openvg")
        sw = ssqt.ShowSegmentationWidget(None, show_load_button=True, show_load_interface=True)
        sw.smoothing = False
        sw.add_data(segmentation, voxelsize_mm=voxelsize_mm)
        # QTest.mouseClick(sw.ui_buttons['Show volume'], Qt.LeftButton)
        # sw.add_vtk_file("~/projects/imtools/mesh.vtk")
        sw.show()
        app.exec_()
        # sw.close()
        # sw.deleteLater()

    @pytest.mark.interactive
    def test_show_donut_with_labels(self):
        """
        creates VTK file from input data
        :return:
        """
        datap = imtools.sample_data.donut()

        segmentation = datap['segmentation']
        voxelsize_mm = datap['voxelsize_mm']
        slab = datap["slab"]
        slab["label 20"] = 20
        slab["label 19"] = 19
        slab["label 18"] = 18
        slab["label 17"] = 17
        slab["label 16"] = 16
        slab["label 15"] = 15
        slab["label 14"] = 14
        slab["label 13"] = 13
        slab["label 12"] = 12
        slab["label 11"] = 11
        slab["label 10"] = 10
        slab["label 9"] = 9
        slab["label 8"] = 8
        slab["label 7"] = 7
        slab["label 6"] = 6
        slab["label 5"] = 5

        import imtools.show_segmentation_qt as ssqt


        app = QApplication(sys.argv)
        # app.setGraphicsSystem("openvg")
        sw = ssqt.ShowSegmentationWidget(None, show_load_button=True, show_load_interface=True)
        sw.smoothing = False
        sw.add_data(segmentation, voxelsize_mm=voxelsize_mm, slab=slab)
        # QTest.mouseClick(sw.ui_buttons['Show volume'], Qt.LeftButton)
        # sw.add_vtk_file("~/projects/imtools/mesh.vtk")
        sw.show()
        app.exec_()

    @pytest.mark.interactive
    def test_show_donut_with_zerosize_label(self):
        """
        creates VTK file from input data
        :return:
        """
        datap = imtools.sample_data.donut()

        segmentation = datap['segmentation']
        voxelsize_mm = datap['voxelsize_mm']

        import imtools.show_segmentation_qt as ssqt


        app = QApplication(sys.argv)
        # app.setGraphicsSystem("openvg")
        sw = ssqt.ShowSegmentationWidget(None, show_load_button=True, show_load_interface=True)
        sw.smoothing = False
        datap["slab"]["empty"]=17
        sw.add_data(segmentation, voxelsize_mm=voxelsize_mm, slab=datap["slab"])
        # QTest.mouseClick(sw.ui_buttons['Show volume'], Qt.LeftButton)
        # sw.add_vtk_file("~/projects/imtools/mesh.vtk")
        sw.show()
        app.exec_()
        # sw.close()
        # sw.deleteLater()

    # @pytest.mark.interactive
    @unittest.skipIf(os.environ.get("TRAVIS", False), "Skip on Travis-CI")
    def test_add_data_and_show(self):
        """
        creates VTK file from input data and show and quit
        :return:
        """
        datap = imtools.sample_data.donut()

        segmentation = datap['segmentation']
        voxelsize_mm = datap['voxelsize_mm']

        import imtools.show_segmentation_qt as ssqt


        app = QApplication(sys.argv)
        # app.setGraphicsSystem("openvg")
        sw = ssqt.ShowSegmentationWidget(None, show_load_button=True)
        sw.smoothing = False
        sw.add_data(segmentation, voxelsize_mm=voxelsize_mm)
        QTest.mouseClick(sw.ui_buttons['Show volume'], Qt.LeftButton)
        # sw.add_vtk_file("~/projects/imtools/mesh.vtk")
        sw.show()
        # app.exec_(exec_)
        sw.close()
        output_vtk_file = sw.vtk_file
        output_vtk_file = sw.get_filename_filled_with_checked_labels("*")
        # sw.
        sw.deleteLater()

        sw = None
        app.quit()
        app.deleteLater()
        # self.qapp.quit()
        # self.qapp.deleteLater()
        # gc.collect()

        # app = QApplication(sys.argv)
        # app.quit()
        # self.qapp.exit()
        # app.exec_()

        output_vtk_file_star = op.abspath(op.expanduser(output_vtk_file))
        filelist = glob.glob(output_vtk_file_star)
        self.assertGreater(len(filelist), 0)

    @pytest.mark.interactive
    def test_select_labels(self):
        """
        creates VTK file from input data
        :return:
        """
        datap = imtools.sample_data.donut()

        segmentation = datap['segmentation']
        voxelsize_mm = datap['voxelsize_mm']
        slab = datap["slab"]
        slab["label 20"] = 20
        slab["label 19"] = 19
        slab["label 18"] = 18
        slab["label 17"] = 17
        slab["label 16"] = 16
        slab["label 15"] = 15
        slab["label 14"] = 14
        slab["label 13"] = 13
        slab["label 12"] = 12
        slab["label 11"] = 11
        slab["label 10"] = 10
        slab["label 9"] = 9
        slab["label 8"] = 8
        slab["label 7"] = 7
        slab["label 6"] = 6
        slab["label 5"] = 5

        import imtools.show_segmentation_qt as ssqt


        app = QApplication(sys.argv)
        # app.setGraphicsSystem("openvg")
        sw = ssqt.SelectLabelWidget(slab=slab, segmentation=segmentation, voxelsize_mm=voxelsize_mm, show_ok_button=True)
        # QTest.mouseClick(sw.ui_buttons['Show volume'], Qt.LeftButton)
        # sw.add_vtk_file("~/projects/imtools/mesh.vtk")
        sw.show()
        app.exec_()

if __name__ == '__main__':
    unittest.main()
