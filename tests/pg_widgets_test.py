#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys


import unittest



from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import pytest



import imtools.sample_data


import imtools.select_label_qt




class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass
        # self.qapp = QApplication(sys.argv)


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
        sw = ssqt.SelectLabelWidget(slab=slab, segmentation=segmentation, voxelsize_mm=voxelsize_mm)
        # QTest.mouseClick(sw.ui_buttons['Show volume'], Qt.LeftButton)
        # sw.add_vtk_file("~/projects/imtools/mesh.vtk")
        sw.show()
        app.exec_()

    @pytest.mark.interactive
    def test_pyqtgraph(self):
        """
        creates VTK file from input data
        :return:
        """
        import pyqtgraph.parametertree as pgpt



        params = [
            {'name': 'Liver', 'type': 'bool', 'value': False, "children": [{"name": "integer", "type": "int", "value": 5}]},
            {'name': 'Porta', 'type': 'bool', 'value': False},
            {'name': 'Basic parameter data types', 'type': 'group', 'children': [
                {'name': 'Integer', 'type': 'int', 'value': 10},
                {'name': 'Float', 'type': 'float', 'value': 10.5, 'step': 0.1},
                {'name': 'String', 'type': 'str', 'value': "hi"},
                {'name': 'List', 'type': 'list', 'values': [1, 2, 3], 'value': 2},
                {'name': 'Named List', 'type': 'list', 'values': {"one": 1, "two": "twosies", "three": [3, 3, 3]},
                 'value': 2},
                {'name': 'Boolean', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
                {'name': 'Color', 'type': 'color', 'value': "FF0", 'tip': "This is a color button"},
                {'name': 'Gradient', 'type': 'colormap'},
                {'name': 'Subgroup', 'type': 'group', 'children': [
                    {'name': 'Sub-param 1', 'type': 'int', 'value': 10},
                    {'name': 'Sub-param 2', 'type': 'float', 'value': 1.2e6},
                ]},
                {'name': 'Text Parameter', 'type': 'text', 'value': 'Some text...'},
                {'name': 'Action Parameter', 'type': 'action'},
            ]},
            {'name': 'Numerical Parameter Options', 'type': 'group', 'children': [
                {'name': 'Units + SI prefix', 'type': 'float', 'value': 1.2e-6, 'step': 1e-6, 'siPrefix': True,
                 'suffix': 'V'},
                {'name': 'Limits (min=7;max=15)', 'type': 'int', 'value': 11, 'limits': (7, 15), 'default': -6},
                {'name': 'DEC stepping', 'type': 'float', 'value': 1.2e6, 'dec': True, 'step': 1, 'siPrefix': True,
                 'suffix': 'Hz'},

            ]},
            {'name': 'Save/Restore functionality', 'type': 'group', 'children': [
                {'name': 'Save State', 'type': 'action'},
                {'name': 'Restore State', 'type': 'action', 'children': [
                    {'name': 'Add missing items', 'type': 'bool', 'value': True},
                    {'name': 'Remove extra items', 'type': 'bool', 'value': True},
                ]},
            ]},
            {'name': 'Extra Parameter Options', 'type': 'group', 'children': [
                {'name': 'Read-only', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz',
                 'readonly': True},
                {'name': 'Renamable', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz',
                 'renamable': True},
                {'name': 'Removable', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz',
                 'removable': True},
            ]},
            # ComplexParameter(name='Custom parameter group (reciprocal values)'),
            # ScalableGroup(name="Expandable Parameter Group", children=[
            #     {'name': 'ScalableParam 1', 'type': 'str', 'value': "default param 1"},
            #     {'name': 'ScalableParam 2', 'type': 'str', 'value': "default param 2"},
            # ]),
        ]

        app = QApplication(sys.argv)
        p = pgpt.Parameter.create(name='params', type='group', children=params)
        t = pgpt.ParameterTree()
        t.setParameters(p)
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

        # import imtools.show_segmentation_qt as ssqt
        # app.setGraphicsSystem("openvg")
        # sw = ssqt.SelectLabelWidget(slab=slab, segmentation=segmentation, voxelsize_mm=voxelsize_mm)
        # QTest.mouseClick(sw.ui_buttons['Show volume'], Qt.LeftButton)
        # sw.add_vtk_file("~/projects/imtools/mesh.vtk")
        # sw.show()
        win = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        win.setLayout(layout)
        layout.addWidget(
            QtWidgets.QLabel("These are two views of the same data. They should always display the same values."), 0, 0, 1,
            2)
        layout.addWidget(t, 1, 0, 1, 1)
        win.show()
        app.exec_()


if __name__ == '__main__':
    unittest.main()
