#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import os.path
# import funkcí z jiného adresáře
import sys

import numpy as np

logger = logging.getLogger(__name__)

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../src/"))

# from pysegbase import pycut
from PyQt4 import QtGui, QtCore

import seededitorqt.plugin

class QtSEdThresholdPlugin(seededitorqt.plugin.QtSEdPlugin):

    def __init__(self):
        super(QtSEdThresholdPlugin, self).__init__()
        self.initUI()
        self.updateUI()

    def initUI(self):
        self.vbox.addWidget(QtGui.QLabel("Threshold segmentation"))
        self.slider_lo_thr = self._create_slider("Low Threshold")
        self.slider_hi_thr = self._create_slider("High Threshold")
        self.slider_open = self._create_slider("Binary Open")
        self.slider_close = self._create_slider("Binary Close")
        self.slider_sigma = self._create_slider("Filter Sigma", connect=self._sigma_changed)
        self.slider_sigma_value = 0.2

        self.runbutton = QtGui.QPushButton("Run")
        self.runbutton.clicked.connect(self.run)

        # self.vbox.addWidget(self.slider_lo_thr)
        self.vbox.addWidget(self.runbutton)

    def _create_slider(self, tooltip=None, connect=None):
        slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        if tooltip is not None:
            slider.setToolTip(tooltip)
        if connect is None:
            connect = self._slider_value_updated
        slider.valueChanged.connect(connect)
        self.vbox.addWidget(slider)
        return slider

    def _sigma_changed(self, slider_value):
        self.slider_sigma_value = slider_value * 0.05
        self._slider_value_updated(self.slider_sigma_value)


    def _slider_value_updated(self, slider_value):
        # slider.value()
        self.showStatus("Slider value: {}".format(slider_value))
        pass

    def updateUI(self):
        if self.data3d is not None:
            self.slider_lo_thr.setRange(np.min(self.data3d), np.max(self.data3d))
            self.slider_hi_thr.setRange(np.min(self.data3d), np.max(self.data3d))
            self.slider_open.setRange(0, 10)
            self.slider_close.setRange(0, 10)
            self.slider_sigma.setRange(0, 100)

    def run(self):
        self.runInit()
        self.segmentation = self.data3d > self.slider_lo_thr.value()
        self.auto_method = ""
        self.fillHoles = True
        self.nObj = 1
        self.get_priority_objects = True
        from imtools.uiThreshold import make_image_processing
        self.imgFiltering, self.threshold = make_image_processing(data=self.data3d, voxelsize_mm=self.voxelsize_mm,
                                                                  seeds_inds=self.seeds,
                                                                  sigma_mm=self.slider_sigma_value,
                                                                  min_threshold=self.slider_lo_thr.value(),
                                                                  max_threshold=self.slider_hi_thr.value(),
                                                                  closeNum=self.slider_close.value(),
                                                                  openNum=self.slider_open.value(),
                                                                  min_threshold_auto_method=self.auto_method,
                                                                  fill_holes=self.fillHoles,
                                                                  get_priority_objects=self.get_priority_objects,
                                                                  nObj=self.nObj)
        self.segmentation = self.imgFiltering
        self.runFinish()
