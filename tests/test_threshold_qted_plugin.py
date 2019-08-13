#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function



import logging


# import funkcí z jiného adresáře
import unittest



logger = logging.getLogger(__name__)

import os.path


import sys



path_to_script = os.path.dirname(os.path.abspath(__file__))
pth = os.path.join(path_to_script, "../../seededitorqt/")
sys.path.insert(0, pth)

import pytest


# from pysegbase import pycut

import seededitorqt


import seededitorqt.plugin


import numpy as np


from PyQt5.QtWidgets import QApplication


class SeedEditorQtTest(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     if sys.version_info.major < 3:
    #         cls.assertCountEqual = cls.assertItemsEqual

    def test_addplugin(self):
        """
        just run editor to see what is new
        Returns:

        """
        app = QApplication(sys.argv)
        data = (np.random.rand(30,31,32) * 100).astype(np.int)
        data[15:40, 13:20, 10:18] += 50
        se = seededitorqt.QTSeedEditor(data)
        wg0 = seededitorqt.plugin.SampleThresholdPlugin()
        se.addPlugin(wg0)
        # se.exec_()
        # self.assertTrue(False)

    @pytest.mark.interactive
    def test_addplugin_interactive(self):
        """
        just run editor to see what is new
        Returns:

        """
        app = QApplication(sys.argv)
        data = (np.random.rand(30,31,32) * 100).astype(np.int)
        data[15:40, 13:20, 10:18] += 50
        se = seededitorqt.QTSeedEditor(data)
        wg0 = seededitorqt.plugin.SampleThresholdPlugin()
        se.addPlugin(wg0)
        # se.exec_()
        # self.assertTrue(False)

    @pytest.mark.interactive
    def test_show_editor(self):
        """
        just run editor to see what is new
        Returns:

        """
        app = QApplication(sys.argv)
        data = (np.random.rand(30,31,32) * 250).astype(np.int)
        data[15:40, 13:20, 10:18] += 150
        se = seededitorqt.QTSeedEditor(data)
        import imtools.threshold_qsed_plugin


        wg0 = imtools.threshold_qsed_plugin.QtSEdThresholdPlugin(debug=True)
        se.addPlugin(wg0)
        se.exec_()
        # self.assertTrue(False)



if __name__ == "__main__":
    unittest.main()
