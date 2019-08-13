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
from PyQt5 import QtCore, QtGui, QtWidgets

import seededitorqt.plugin

class QtSEdThresholdPlugin(seededitorqt.plugin.QtSEdPlugin):
    def __init__(
        self,
        threshold=None,
        filter_sigma=0.2,
        nObj=1,
        biggestObjects=True,
        useSeedsOfCompactObjects=True,
        binaryClosingIterations=2,
        binaryOpeningIterations=0,
        fillHoles=True,
        threshold_auto_method="",
        threshold_upper=None,
        debug=False,
    ):
        """

        Inicialitacni metoda.
        Input:
            :param data: data pro prahovani, se kterymi se pracuje
            :param voxel: velikost voxelu
            :param threshold:
            :param interactivity: zapnuti / vypnuti gui
            :param number: maximalni hodnota slideru pro gauss. filtrovani (max sigma)
            :param inputSigma: pocatecni hodnota pro gauss. filtr
            :param nObj: pocet nejvetsich objektu k vraceni
            :param biggestObjects: oznacuje, zda se maji vracet nejvetsi objekty
            :param binaryClosingIterations: iterace binary closing
            :param binaryOpeningIterations: iterace binary opening
            :param seeds: matice s kliknutim uzivatele- pokud se maji vracet
                   specifikce objekty. It can be same shape like data, or it can be
                   indexes e.g. from np.nonzero(seeds)
            :param cmap: grey
            :param threshold_auto_method: 'otsu' use otsu threshold, other string use our liver automatic

        """
        super(QtSEdThresholdPlugin, self).__init__()

        logger.debug("Spoustim prahovani dat...")
        self.on_close_fcn = None

        self.errorsOccured = False
        self.filter_sigma = filter_sigma
        # if shapes of input data and seeds are the same

        # import ipdb; ipdb.set_trace()
        self.threshold = threshold
        self.nObj = nObj
        self.biggestObjects = biggestObjects
        self.ICBinaryClosingIterations = binaryClosingIterations
        self.ICBinaryOpeningIterations = binaryOpeningIterations
        self.threshold_auto_method = threshold_auto_method

        self.useSeedsOfCompactObjects = useSeedsOfCompactObjects
        self.fillHoles = fillHoles

        self.threshold_upper = threshold_upper

        # Kalkulace objemove jednotky (voxel) (V = a*b*c)
        # voxel1 = self.voxel[0]
        # voxel2 = self.voxel[1]
        # voxel3 = self.voxel[2]
        self.voxelV = np.prod(self.voxelsize_mm, axis=None)  # voxel1 * voxel2 * voxel3

        if self.biggestObjects == True or (
            self.seeds != None and self.useSeedsOfCompactObjects
        ):
            self.get_priority_objects = True
        else:
            self.get_priority_objects = True
        self.debug = debug

        self.initUI()
        self.updateUI()

    def initUI(self):
        self.vbox.addWidget(QtWidgets.QLabel("Threshold segmentation"))
        self.slider_lo_thr = self._create_slider("Low Threshold")
        self.slider_hi_thr = self._create_slider("High Threshold")
        self.slider_open = self._create_slider("Binary Open")
        self.slider_close = self._create_slider("Binary Close")
        self.slider_sigma = self._create_slider(
            "Filter Sigma", connect=self._sigma_changed
        )
        self.slider_sigma_value = 0.2

        self.runbutton = QtWidgets.QPushButton("Run")
        self.runbutton.clicked.connect(self.run)

        # self.vbox.addWidget(self.slider_lo_thr)
        self.vbox.addWidget(self.runbutton)

    def _create_slider(self, tooltip=None, connect=None):
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
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
            # if self.threshold is None and self.seeds is not None:
            #     from imtools.uiThreshold import prepare_threshold_from_seeds
            #     threshold = prepare_threshold_from_seeds(
            #         data=self.data3d, seeds=self.seeds, min_threshold_auto_method=self.threshold_auto_method)
            #     self.threshold = threshold
            #     self.slider_lo_thr.setValue(int(self.threshold))
            # logger.debug("threshold after first evaluation {}".format(threshold))
            self.slider_lo_thr.setRange(np.min(self.data3d), np.max(self.data3d))
            self.slider_hi_thr.setRange(np.min(self.data3d), np.max(self.data3d))
            self.slider_hi_thr.setValue(np.max(self.data3d))
            self.slider_open.setRange(0, 10)
            self.slider_close.setRange(0, 10)
            self.slider_sigma.setRange(0, 100)

    def run(self):
        self.runInit()

        from imtools.uiThreshold import make_image_processing



        self.imgFiltering, self.threshold = make_image_processing(
            data=self.data3d,
            voxelsize_mm=self.voxelsize_mm,
            seeds=self.seeds,
            sigma_mm=self.slider_sigma_value,
            min_threshold=self.slider_lo_thr.value(),
            max_threshold=self.slider_hi_thr.value(),
            closeNum=self.slider_close.value(),
            openNum=self.slider_open.value(),
            min_threshold_auto_method=self.threshold_auto_method,
            fill_holes=self.fillHoles,
            get_priority_objects=self.get_priority_objects,
            nObj=self.nObj,
            debug=self.debug
        )
        self.segmentation = self.imgFiltering
        self.runFinish()
