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
from PyQt5.QtWidgets import (QGridLayout, QPushButton, QCheckBox, QWidget,
                         QVBoxLayout)
# from PyQt5 import QtGui, QtWidgets
from PyQt5 import QtCore, QtWidgets
import numpy as np
# import pyqtgraph as pg


class SelectLabelWidget(QtWidgets.QWidget):
    def __init__(self, slab=None, segmentation=None, voxelsize_mm=None, show_ok_button=True, app=None):
        super(SelectLabelWidget, self).__init__()
        self.app = app
        self.ui_slab = {}

        self.mainLayout = QGridLayout(self)
        widget = QWidget()
        widget.setLayout(self.mainLayout)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(400)
        scroll.setWidget(widget)

        self.superMainScrollLayout = QVBoxLayout(self)
        self.superMainScrollLayout.addWidget(scroll)
        self.setLayout(self.superMainScrollLayout)

        # scroll.setLayout(self.mainLayout)

        self.init_slab(slab=slab, segmentation=segmentation, voxelsize_mm=voxelsize_mm, show_ok_button=show_ok_button)
        self.update_slab_ui()

    def init_slab(self, slab=None, segmentation=None, voxelsize_mm=None, show_ok_button=False):
        """
        Create widget with segmentation labels information used to select labels.

        :param slab: dict with label name and its id
        :param segmentation: 3D label ndarray
        :param voxelsize_mm: size of voxel in mm
        :return:
        """
        self.segmentation = segmentation
        self.voxelsize_mm = voxelsize_mm

        from . import show_segmentation

        self.slab = show_segmentation.create_slab_from_segmentation(
            self.segmentation, slab=slab)

        if show_ok_button:
            ok_button = QPushButton("Ok")
            ok_button.clicked.connect(self._action_ok_button)

            self.superMainScrollLayout.addWidget(ok_button)

    def _action_ok_button(self):
        self.close()

    def update_slab_ui(self):
        _row = 1
        # remove old widgets
        for key, val in self.ui_slab.items():
            val.deleteLater()
        self.ui_slab = {}
        # _row_slab = 0
        for label, value in self.slab.items():
            _row += 1
            # _row_slab += 1
            txt = str(label) + "(" + str(value) + "): "
            nvoxels = 0
            if self.segmentation is not None:
                nvoxels = np.sum(self.segmentation==value)
                if self.voxelsize_mm is not None:
                    vx_vol = np.prod(self.voxelsize_mm)
                    txt += str(nvoxels * vx_vol) + " [mm^3], "
                txt += str(nvoxels)
            self.ui_slab[label] = QCheckBox(txt, self)
            self.mainLayout.addWidget(self.ui_slab[label], _row, 1, 1, 2)
            if value != 0 and nvoxels > 0:
                self.ui_slab[label].setCheckState(QtCore.Qt.Checked)
                # self.ui_buttons["Show"].clicked.connect(self.actionShow)

        return _row

    def action_check_slab_ui(self):
        return self.get_selected_labels()

    def get_selected_labels(self):
        labels = []
        for key, val in self.ui_slab.items():
            if val.isChecked():
                labels.append(self.slab[key])

        return labels

    def run(self):

        self.app.exec_()
        return self.get_selected_labels()


class SelectLabelWidget2(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super(SelectLabelWidget, self).__init__()
        pass

