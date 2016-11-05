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
from PyQt4.QtGui import QGridLayout, QLabel, QPushButton, QLineEdit, QCheckBox
from PyQt4 import QtGui
from PyQt4 import QtCore
import sys
import numpy as np


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
        lblSegConfig = QLabel('Choose configure')
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
        self.ui_buttons["Show"] = QPushButton("Show", self)
        self.ui_buttons["Show"].clicked.connect(self.actionShow)
        self.mainLayout.addWidget(self.ui_buttons['Show'], self._row, 1, 1, 6)

    def action_ui_params(self):
        self.resize_mm = str(self.ui_buttons['resize_mm'].text())
        if self.resize_mm == "None":
            self.resize_mm = None
        else:
            self.resize_mm = float(self.resize_mm)

        self.smoothing = self.ui_buttons['smoothing'].isChecked()


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
        ds = show_segmentation.select_labels(self.segmentation, labels)

        show_segmentation.showSegmentation(
            # self.segmentation,
            ds,
            degrad=self.degrad,
            voxelsize_mm=self.voxelsize_mm,
            vtk_file=self.vtk_file,
            resize_mm=self.resize_mm,
            smoothing=self.smoothing
        )

    def initLabels(self):
        btnHearth = QPushButton("Hearth", self)
        btnHearth.setCheckable(True)
        btnHearth.clicked.connect(self.configEvent)
        self.mainLayout.addWidget(btnHearth, 2, 1)

        btnKidneyL = QPushButton("Kidney Left", self)
        btnKidneyL.setCheckable(True)
        btnKidneyL.clicked.connect(self.configEvent)
        self.mainLayout.addWidget(btnKidneyL, 2, 2)

        btnKidneyR = QPushButton("Kidney Right", self)
        btnKidneyR.setCheckable(True)
        btnKidneyR.clicked.connect(self.configEvent)
        self.mainLayout.addWidget(btnKidneyR, 2, 3)

        btnLiver = QPushButton("Liver", self)
        btnLiver.setCheckable(True)
        btnLiver.clicked.connect(self.configEvent)
        self.mainLayout.addWidget(btnLiver, 2, 4)

        self.group = QtGui.QButtonGroup()
        self.group.addButton(btnHearth)
        self.group.addButton(btnKidneyL)
        self.group.addButton(btnKidneyR)
        self.group.addButton(btnLiver)
        self.group.setId(btnHearth, 1)
        self.group.setId(btnKidneyL, 2)
        self.group.setId(btnKidneyR, 3)
        self.group.setId(btnLiver, 4)

    def initLabelsAuto(self):
        position = 1
        self.groupA = QtGui.QButtonGroup()
        for key, value in self.oseg.slab.items():
            btnLabel = QPushButton(key)
            btnLabel.setCheckable(True)
            btnLabel.clicked.connect(self.configAutoEvent)
            self.mainLayout.addWidget(btnLabel, 12, position)
            self.groupA.addButton(btnLabel)
            self.groupA.setId(btnLabel, position)
            position += 1

    def configAutoEvent(self):
        alt_seg_params = {
            "output_label": 'left kidney',
            'clean_seeds_after_update_parameters': True,
        }
        id = self.groupA.checkedId()
        print id
        selected_label = self.oseg.slab.keys()[id - 1]
        alt_seg_params['output_label'] = selected_label
        self.oseg.update_parameters(alt_seg_params)

    def initConfigs(self):
        self.btnSegManual = QPushButton("Manual", self)
        # btnSegManual.clicked.connect(self.btnManualSeg)
        self.mainLayout.addWidget(self.btnSegManual, 6, 1)

        self.btnSegSemiAuto = QPushButton("Semi-automatic", self)
        # btnSegSemiAuto.clicked.connect(self.btnSemiautoSeg)
        self.mainLayout.addWidget(self.btnSegSemiAuto, 6, 2)

        self.btnSegMask = QPushButton("Mask", self)
        # btnSegMask.clicked.connect(self.maskRegion)
        self.mainLayout.addWidget(self.btnSegMask, 6, 3)

        self.btnSegPV = QPushButton("Portal Vein", self)
        # btnSegPV.clicked.connect(self.btnPortalVeinSegmentation)
        self.mainLayout.addWidget(self.btnSegPV, 6, 4)

        self.btnSegHV = QPushButton("Hepatic Vein", self)
        # btnSegHV.clicked.connect(self.btnHepaticVeinsSegmentation)
        self.mainLayout.addWidget(self.btnSegHV, 6, 5)

        self.disableSegType()

    def configEvent(self, event):
        id = self.group.checkedId()
        self.lblSegError.setText("")
        if id == 1:
            self.oseg.update_parameters_based_on_label("label hearth")
            self.enableSegType()
        elif id == 2:
            self.oseg.update_parameters_based_on_label("label kidney L")
            self.enableSegType()
        elif id == 3:
            self.oseg.update_parameters_based_on_label("label kidney R")
            self.enableSegType()
        elif id == 4:
            self.oseg.update_parameters_based_on_label("label liver")
            self.enableSegType()
        else:
            self.lblSegError.setText("Unknown error: Config have not been set.")
            self.disableSegType()

    def enableSegType(self):
        self.btnSegManual.setDisabled(False)
        self.btnSegSemiAuto.setDisabled(False)
        self.btnSegMask.setDisabled(False)
        self.btnSegPV.setDisabled(False)
        self.btnSegHV.setDisabled(False)

    def disableSegType(self):
        self.btnSegManual.setDisabled(True)
        self.btnSegSemiAuto.setDisabled(True)
        self.btnSegMask.setDisabled(True)
        self.btnSegPV.setDisabled(True)
        self.btnSegHV.setDisabled(True)


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
