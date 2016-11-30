#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from nose.plugins.attrib import attr

import sys

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

    # @attr('interactive')
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

if __name__ == '__main__':
    unittest.main()
