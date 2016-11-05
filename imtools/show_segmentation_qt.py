#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © %YEAR%  <>
#
# Distributed under terms of the %LICENSE% license.

"""

"""

import logging

logger = logging.getLogger(__name__)
import argparse
from PyQt4.QtGui import QGridLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QFileDialog
from PyQt4 import QtGui
from PyQt4 import QtCore
import sys
import numpy as np

import vtk
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class ShowSegmentationWidget(QtGui.QWidget):
    def __init__(self, segmentation, voxelsize_mm=[1,1,1], slab=None, **nargs):
        super(ShowSegmentationWidget, self).__init__()
        # self.data3d = data3d
        self.segmentation = segmentation
        self.voxelsize_mm = np.asarray(voxelsize_mm)
        self.slab = slab
        self.resize_mm = None
        self.degrad = 6
        self.smoothing=True
        self.vtk_file = "mesh.vtk"
        self.ui_buttons = {}

        self.init_slab(slab)

        self.initUI()



    def initUI(self):

        self.mainLayout = QGridLayout(self)

        self._row = 0
        lblSegConfig = QLabel('Labels')
        self.mainLayout.addWidget(lblSegConfig, self._row, 1, 1, 6)

        self.init_slab_ui()

        self._row += 1
        resizeQLabel= QLabel('resize_mm')
        self.mainLayout.addWidget(resizeQLabel, self._row, 1)

        self.ui_buttons["resize_mm"] = QLineEdit()
        self.ui_buttons["resize_mm"].setText(str(self.resize_mm))
        self.mainLayout.addWidget(self.ui_buttons['resize_mm'], self._row, 2)

        self._row += 1
        keyword = "smoothing"
        # smoothingQLabel= QLabel(keyword)
        # self.mainLayout.addWidget(smoothingQLabel, self._row, 1)

        self.ui_buttons[keyword] = QCheckBox(keyword, self)
        # self.ui_buttons[keyword].setTristate(False)
        if self.smoothing:
            sm_state = QtCore.Qt.Checked
        else:
            sm_state =  QtCore.Qt.Unchecked
        self.ui_buttons[keyword].setCheckState(sm_state)
        self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 1)


        self._row += 1
        keyword = "vtk_file"
        vtk_fileQLabel= QLabel("Output VTK file")
        self.mainLayout.addWidget(vtk_fileQLabel, self._row, 1)

        self.ui_buttons[keyword] = QLineEdit()
        self.ui_buttons[keyword].setText(str(self.vtk_file))
        self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 2)
        keyword = "vkt_file_button"
        self.ui_buttons[keyword] = QPushButton("Set", self)
        self.ui_buttons[keyword].clicked.connect(self.action_select_vtk_file)
        self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 3)

        self._row += 1
        keyword = "Show"
        self.ui_buttons[keyword] = QPushButton(keyword, self)
        self.ui_buttons[keyword].clicked.connect(self.actionShow)
        self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 1, 1, 3)

        self._row += 1
        keyword = "Add extern file"
        self.ui_buttons[keyword] = QPushButton(keyword, self)
        self.ui_buttons[keyword].clicked.connect(self.action_add_file)
        self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 1, 1, 3)

        # vtk + pyqt
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.mainLayout.addWidget(self.vtkWidget, 0, 4, self._row + 1, 1)
        self.renWin = self.vtkWidget.GetRenderWindow()
        self.renWin.AddRenderer(self.renderer)


        # vtkviewer
        import vtkviewer
        self.vtkv = vtkviewer.VTKViewer()
        self.vtkv.renderer = self.renderer
        self.vtkv.iren = self.vtkWidget
        self.vtkv.renWin = self.renWin
        self.vtkWidget.resize(300,300)
        self.updateGeometry()
        self.vtkv_started = False



    def vtkv_start(self):
        if not self.vtkv_started:
            self.vtkv.Start()


    def action_select_vtk_file(self):
        self.ui_buttons["vtk_file"].setText(QFileDialog.getSaveFileName())

    def action_add_file(self):
        self.vtkv.AddFile(str(QFileDialog.getOpenFileName()))
        self.vtkv_start()

        # self.vtkv.Start()
        # or show win
        # self.renWin.Render()
        # self.vtkv.iren.Start()





    def action_ui_params(self):
        self.resize_mm = str(self.ui_buttons['resize_mm'].text())
        if self.resize_mm == "None":
            self.resize_mm = None
        else:
            self.resize_mm = float(self.resize_mm)

        self.smoothing = self.ui_buttons['smoothing'].isChecked()

        self.vtk_file = str(self.ui_buttons["vtk_file"].text())

    def init_slab(self, slab):

        if slab is None:
            labels = np.unique(self.segmentation)
            slab = {}
            for label in labels:
                slab[label] = label
        self.slab = slab

    def init_slab_ui(self):
        self.ui_slab = {}
        for label, value in self.slab.iteritems():
            self._row += 1
            self.ui_slab[label] = QCheckBox(str(label) + "(" + str(value) + ")", self)
            self.mainLayout.addWidget(self.ui_slab[label], self._row, 1, 1, 6)
            if value != 0:
                self.ui_slab[label].setCheckState(QtCore.Qt.Checked)
            # self.ui_buttons["Show"].clicked.connect(self.actionShow)



        pass

    def action_check_slab_ui(self):
        labels = []
        for key, val in self.ui_slab.iteritems():
            if val.isChecked():
                labels.append(self.slab[key])

        return labels



    def actionShow(self):
        import show_segmentation
        labels = self.action_check_slab_ui()
        self.action_ui_params()
        ds = show_segmentation.select_labels(self.segmentation, labels)

        show_segmentation.showSegmentation(
            # self.segmentation,
            ds,
            degrad=self.degrad,
            voxelsize_mm=self.voxelsize_mm,
            vtk_file=self.vtk_file,
            resize_mm=self.resize_mm,
            smoothing=self.smoothing,
            show=False
        )

        # self._run_viewer()
        self.vtkv.AddFile(self.vtk_file)
        self.vtkv.Start()




def main():
    import io3d
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # create file handler which logs even debug messages
    # fh = logging.FileHandler('log.txt')
    # fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.debug('start')

    # input parser
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        '-i', '--inputfile',
        default=None,
        required=True,
        help='input file'
    )
    parser.add_argument(
        '--dict',
        default="{'jatra':2, 'ledviny':7}",
        # required=True,
        help='input dict'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)

    app = QtGui.QApplication(sys.argv)

    # w = QtGui.QWidget()
    # w = DictEdit(dictionary={'jatra':2, 'ledviny':7})
    datap = io3d.read(args.inputfile, dataplus_format=True)
    if not 'segmentation' in datap.keys():
        datap['segmentation'] = datap['data3d']

    w = ShowSegmentationWidget(**datap)
    w.resize(250, 150)
    w.move(300, 300)
    w.setWindowTitle('Simple')
    w.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
