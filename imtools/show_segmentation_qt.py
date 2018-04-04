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
import io3d.outputqt
from . import image_manipulation as imma

class SelectLabelWidget(QtGui.QWidget):
    def __init__(self, *args, **kwargs):
        super(SelectLabelWidget, self).__init__()
        self.ui_slab = {}

        self.mainLayout = QGridLayout(self)
        self.init_slab(*args, **kwargs)
        self.update_slab_ui()

    def init_slab(self, slab=None, segmentation=None, voxelsize_mm=None):
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

    def update_slab_ui(self):
        _row = 1
        # remove old widgets
        for key, val in self.ui_slab.iteritems():
            val.deleteLater()
        self.ui_slab = {}
        # _row_slab = 0
        for label, value in self.slab.iteritems():
            _row += 1
            # _row_slab += 1
            txt = str(label) + "(" + str(value) + "): "
            if self.segmentation is not None:
                nvoxels =  np.sum(self.segmentation==value)
                if self.voxelsize_mm is not None:
                    vx_vol = np.prod(self.voxelsize_mm)
                    txt += str(nvoxels * vx_vol) + " [mm^3], "
                txt += str(nvoxels)
            self.ui_slab[label] = QCheckBox(txt, self)
            self.mainLayout.addWidget(self.ui_slab[label], _row, 1, 1, 2)
            if value != 0:
                self.ui_slab[label].setCheckState(QtCore.Qt.Checked)
                # self.ui_buttons["Show"].clicked.connect(self.actionShow)

        return _row



        pass

    def action_check_slab_ui(self):
        labels = []
        for key, val in self.ui_slab.iteritems():
            if val.isChecked():
                labels.append(self.slab[key])

        return labels


class ShowSegmentationWidget(QtGui.QWidget):
    def __init__(self, segmentation=None, vtk_file="mesh_{}.vtk", *args, **kwargs):
        super(ShowSegmentationWidget, self).__init__()
        self.ui_buttons = {}

        self.show_load_interface = False
        if "show_load_button" in kwargs:
            self.show_load_interface = kwargs.pop("show_load_button")
        if "show_load_interface" in kwargs:
            self.show_load_interface = kwargs.pop("show_load_interface")

        self.segmentation = None
        self.init_parameters()
        # self.slab_wg =
        # self.init_slab()
        self.vtk_file = vtk_file
        self._init_ui()
        if segmentation is not None:
            logger.debug("segmentation is not none")
            self.add_data(segmentation, *args, **kwargs)

    # def init_slab(self):
    #     self.

    def add_data_file(self, filename):
        import io3d
        datap = io3d.read(filename, dataplus_format=True)
        if not 'segmentation' in datap.keys():
            datap['segmentation'] = datap['data3d']

        self.add_data(**datap)

    def init_parameters(self):
        self.resize_mm = None
        self.resize_voxel_number = 10000
        self.degrad = 1
        self.smoothing=True

    def add_data(self, segmentation, voxelsize_mm=[1,1,1], slab=None, **kwargs):
        # self.data3d = data3d
        self.segmentation = segmentation
        self.voxelsize_mm = np.asarray(voxelsize_mm)
        self.slab = slab
        self.init_parameters()

        self.init_slab(slab)

        self.update_slab_ui()

    def init_slab(self, slab=None):
        """

        :param slab:
        :return:
        """

        self.slab_wg.init_slab(slab=slab, segmentation=self.segmentation, voxelsize_mm=self.voxelsize_mm)

    def update_slab_ui(self):
        self.slab_wg.update_slab_ui()

    def _init_ui(self):

        self.mainLayout = QGridLayout(self)

        self._row = 0

        if self.show_load_interface:
            keyword = "add_data_file"
            self.ui_buttons[keyword] = QPushButton("Load volumetric data", self)
            self.ui_buttons[keyword].clicked.connect(self._ui_callback_add_data_file)
            self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 1, 1, 3)

            self._row += 1
            lblSegConfig = QLabel('Labels')
            self.mainLayout.addWidget(lblSegConfig, self._row, 1, 1, 6)

        # self.slab_widget = QGridLayout(self)
        # self.mainLayout.addWidget(self.slab_widget, self._row, 1)

            self._row += 1
            self.slab_wg = SelectLabelWidget()
            self.slab_wg.init_slab()
            self.slab_wg.update_slab_ui()
            self.mainLayout.addWidget(self.slab_wg, self._row, 1, 1, 3)
            # self._row = self.update_slab_ui()

            self._row = 10
            self._row += 1
            keyword = "resize_voxel_number"
            resizeQLabel= QLabel('Resize [voxel number]')
            self.mainLayout.addWidget(resizeQLabel, self._row, 1)

            self.ui_buttons[keyword] = QLineEdit()
            self.ui_buttons[keyword].setText(str(self.resize_voxel_number))
            self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 2)

            self._row += 1
            keyword = "resize_mm"
            resizeQLabel= QLabel('Resize [mm]')
            self.mainLayout.addWidget(resizeQLabel, self._row, 1)
            self.mainLayout.setColumnMinimumWidth(2, 100)

            self.ui_buttons[keyword] = QLineEdit()
            self.ui_buttons[keyword].setText(str(self.resize_mm))
            self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 2)

            self._row += 1
            keyword = "degrad"
            resizeQLabel= QLabel('Degradation')
            self.mainLayout.addWidget(resizeQLabel, self._row, 1)

            self.ui_buttons[keyword] = QLineEdit()
            self.ui_buttons[keyword].setText(str(self.degrad))
            self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 2)

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
            sopw = io3d.outputqt.SelectOutputPathWidget(path=self.vtk_file, parent=self)
            self.ui_buttons["output file"] = sopw
            self.mainLayout.addWidget(sopw, self._row, 1, 1, 3)
            sopw.show()

            # keyword = "vtk_file"
            # vtk_fileQLabel= QLabel("Output VTK file")
            # self.mainLayout.addWidget(vtk_fileQLabel, self._row, 1)
            #
            # self.ui_buttons[keyword] = QLineEdit()
            # self.ui_buttons[keyword].setText(str(self.vtk_file))
            # self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 2)
            # keyword = "vkt_file_button"
            # self.ui_buttons[keyword] = QPushButton("Set", self)
            # self.ui_buttons[keyword].clicked.connect(self.action_select_vtk_file)
            # self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 3)

            self._row += 1
            keyword = "Show volume"
            self.ui_buttons[keyword] = QPushButton(keyword, self)
            self.ui_buttons[keyword].clicked.connect(self.actionShow)
            self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 1, 1, 3)

            self._row += 1
            keyword = "Save each"
            self.ui_buttons[keyword] = QPushButton(keyword, self)
            self.ui_buttons[keyword].clicked.connect(self._action_save_all)
            self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 1, 1, 3)

            self._row += 1
            keyword = "Add extern file"
            self.ui_buttons[keyword] = QPushButton(keyword, self)
            self.ui_buttons[keyword].clicked.connect(self._ui_action_add_vtk_file)
            self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 1, 1, 3)

            self._row += 1
            keyword = "Clear"
            self.ui_buttons[keyword] = QPushButton(keyword, self)
            self.ui_buttons[keyword].clicked.connect(self.clear_3d_viewer)
            self.mainLayout.addWidget(self.ui_buttons[keyword], self._row, 1, 1, 3)

        # vtk + pyqt
        self._viewer_height = self._row
        self._init_3d_viewer()


    def _init_3d_viewer(self):
        logger.debug("init 3d view")
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.mainLayout.addWidget(self.vtkWidget, 0, 4, self._viewer_height + 1, 1)
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

    def _ui_callback_add_data_file(self):
        self.add_data_file(str(QFileDialog.getOpenFileName()))

    def add_vtk_file(self, filename):
        self.vtkv.AddFile(filename)
        self.vtkv_start()

    def add_vtk_polydata(self, polydata, colormap=None):
        self.vtkv.AddPolyData(polydata, colormap)
        self.vtkv_start()

    def _ui_action_add_vtk_file(self):
        fn = str(QFileDialog.getOpenFileName())
        import os.path as op
        ext = op.splitext(fn)[1]
        if ext == ".yaml":
            import gen_vtk_tree
            polydata = gen_vtk_tree.vt_file2polyData(fn)
            self.add_vtk_polydata(polydata)
        else:
            self.add_vtk_file(fn)

        # self.vtkv.Start()
        # or show win
        # self.renWin.Render()
        # self.vtkv.iren.Start()

    def clear_3d_viewer(self):
        ren = self.renderer
        actors = self.renderer.GetActors()
        act = actors.GetLastActor()
        while act is not None:
            ren.RemoveActor(act)
            act = actors.GetLastActor()

        # self.renderer.removeAllViewProps()
        # self.renderer = None
        # self.mainLayout.removeWidget(self.vtkWidget)
        # self.vtkWidget.deleteLater()
        # self.vtkWidget = None
        # self.renWin = None
        # self._init_3d_viewer()
        # self.vtkWidget.show()

    def _find_None(self, lineedit):
        text = str(lineedit.text())

        print("find none ", text)
        if text == "None":
            text = None
        else:
            text = float(text)

        return text


    def action_ui_params(self):
        self.resize_mm = self._find_None(self.ui_buttons['resize_mm'])
        self.resize_voxel_number = self._find_None(self.ui_buttons['resize_voxel_number'])
        self.degrad = self._find_None(self.ui_buttons['degrad'])
        self.smoothing = self.ui_buttons['smoothing'].isChecked()
        self.vtk_file = str(self.ui_buttons["output file"].get_path())

        # print("degrad", self.degrad)


    def _action_save_all(self):
        # slab = self.slab
        labels = self.slab_wg.action_check_slab_ui()
        self.action_ui_params()
        # imma.get_nlabel()

        # for lab in labels:
        #     # labi = slab[lab]
        #     strlabel = imma.get_nlabels(slab=self.slab_wg.slab, labels=lab, return_mode="str")
        #     logger.debug(strlabel)
        #     filename = self.vtk_file.format(strlabel)
        #     logger.debug(filename)
        self.show_labels(
            labels,
            self.vtk_file,
            together_vtk_file=False
        )


    def get_filename_filled_with_checked_labels(self, labels=None):
        """ Fill used labels into filename """
        if labels is None:
            labels = self.slab_wg.action_check_slab_ui()
        string_labels = imma.get_nlabels(slab=self.slab_wg.slab, labels=labels, return_mode="str")
        filename = self.vtk_file.format(
            "-".join(string_labels))
        return filename

    def actionShow(self):
        logger.debug("actionShow")
        labels = self.slab_wg.action_check_slab_ui()
        self.action_ui_params()
        filename = self.get_filename_filled_with_checked_labels(labels)
        self.show_labels(labels, filename)

    def show_labels(self, labels, vtk_file, together_vtk_file=True):
        from . import show_segmentation
        # ds = show_segmentation.select_labels(self.segmentation, labels, slab=self.slab_wg.slab)
        # if ds.max() == False:
        #     logger.info("Nothing found for labels " + str(labels))
        #     return
        if together_vtk_file:
            prepare_fcn = show_segmentation.prepare_vtk_file
        else:
            prepare_fcn = show_segmentation.prepare_vtk_files

        vtk_files = prepare_fcn(
            # self.segmentation,
            self.segmentation,
            degrad=self.degrad,
            voxelsize_mm=self.voxelsize_mm,
            vtk_file=vtk_file,
            labels=labels,
            resize_mm=self.resize_mm,
            resize_voxel_number=self.resize_voxel_number,
            smoothing=self.smoothing,
            slab=self.slab_wg.slab
        )

        # self._run_viewer()
        if together_vtk_file:
            self.vtkv.AddFile(vtk_file)
            self.vtkv.Start()
        else:
            for vtkf in vtk_files:
                self.vtkv.AddFile(vtkf)
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
        # required=True,
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
    if args.inputfile is None:
        datap = {
            'segmentation': None
        }
    else:
        datap = io3d.read(args.inputfile, dataplus_format=True)
        if not 'segmentation' in datap.keys():
            datap['segmentation'] = datap['data3d']

    # import ipdb; ipdb.set_trace()
    w = ShowSegmentationWidget(show_load_button=True,**datap)
    # w = SelectLabelWidget(segmentation=datap["segmentation"])
    w.resize(250, 150)
    w.move(300, 300)
    w.setWindowTitle('Simple')
    w.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
