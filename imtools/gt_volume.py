#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Generator of histology report

"""
import logging
logger = logging.getLogger(__name__)

import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))


import argparse
import numpy as np
import scipy.ndimage
import misc

# import datareader
import sed3 as se

import gen_vtk_tree

import vtk
from vtk.util import numpy_support

from datetime import datetime


class VolumeTreeGenerator:
    """
    This generator is called by generateTree() function as a general form.
    Other similar generator is used for generating LAR outputs.
    """
    def __init__(self, gtree):
        self.shape = gtree.shape
        self.data3d = np.zeros(gtree.shape, dtype=np.int)
        self.voxelsize_mm = gtree.voxelsize_mm

    def add_cylinder(self, p1m, p2m, rad, id):
        """
        Funkce na vykresleni jednoho segmentu do 3D dat
        """

        cyl_data3d = np.ones(self.shape, dtype=np.bool)
        # prvni a koncovy bod, ve pixelech
        p1 = [p1m[0] / self.voxelsize_mm[0], p1m[1] /
              self.voxelsize_mm[1], p1m[2] / self.voxelsize_mm[2]]
        p2 = [p2m[0] / self.voxelsize_mm[0], p2m[1] /
              self.voxelsize_mm[1], p2m[2] / self.voxelsize_mm[2]]
        logger.debug(
            "p1_px: " + str(p1[0]) + " " + str(p1[1]) + " " + str(p1[2]))
        logger.debug(
            "p2_px: " + str(p2[0]) + " " + str(p2[1]) + " " + str(p2[2]))
        logger.debug("radius_mm:" + str(rad))

        # vzdalenosti mezi prvnim a koncovim bodem (pro jednotlive osy)
        pdiff = [abs(p1[0] - p2[0]), abs(p1[1] - p2[1]), abs(p1[2] - p2[2])]

        # generovani hodnot pro osu segmentu
        num_points = max(pdiff) * \
            2  # na jeden "pixel nejdelsi osy" je 2 bodu primky (shannon)
        zvalues = np.linspace(p1[0], p2[0], num_points)
        yvalues = np.linspace(p1[1], p2[1], num_points)
        xvalues = np.linspace(p1[2], p2[2], num_points)

        # drawing a line
        for i in range(0, len(xvalues)):
            try:
                cyl_data3d[int(zvalues[i])][int(yvalues[i])][int(xvalues[i])] = 0
            except:
                import traceback
                traceback.print_exc()
                print "except in drawing line"
                import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        # cuting size of 3d space needed for calculating distances (smaller ==
        # a lot faster)
        cut_up = max(
            0, round(min(p1[0], p2[0]) - (rad / min(self.voxelsize_mm)) - 2))
        # ta 2 je kuli tomu abyh omylem nurizl
        cut_down = min(self.shape[0], round(
            max(p1[0], p2[0]) + (rad / min(self.voxelsize_mm)) + 2))
        cut_yu = max(
            0, round(min(p1[1], p2[1]) - (rad / min(self.voxelsize_mm)) - 2))
        cut_yd = min(self.shape[1], round(
            max(p1[1], p2[1]) + (rad / min(self.voxelsize_mm)) + 2))
        cut_xl = max(
            0, round(min(p1[2], p2[2]) - (rad / min(self.voxelsize_mm)) - 2))
        cut_xr = min(self.shape[2], round(
            max(p1[2], p2[2]) + (rad / min(self.voxelsize_mm)) + 2))
        logger.debug("cutter_px: z_up-" + str(cut_up) + " z_down-" + str(cut_down) + " y_up-" + str(
            cut_yu) + " y_down-" + str(cut_yd) + " x_left-" + str(cut_xl) + " x_right-" + str(cut_xr))
        cyl_data3d_cut = cyl_data3d[
            int(cut_up):int(cut_down),
            int(cut_yu):int(cut_yd),
            int(cut_xl):int(cut_xr)]

        # calculating distances
        # spotrebovava naprostou vetsinu casu (pro 200^3  je to kolem 1.2
        # sekundy, proto jsou data osekana)
        lineDst = scipy.ndimage.distance_transform_edt(
            cyl_data3d_cut, self.voxelsize_mm)

        # zkopirovani vyrezu zpet do celeho rozsahu dat
        for z in xrange(0, len(cyl_data3d_cut)):
            for y in xrange(0, len(cyl_data3d_cut[z])):
                for x in xrange(0, len(cyl_data3d_cut[z][y])):
                    if lineDst[z][y][x] <= rad:
                        iX = int(z + cut_up)
                        iY = int(y + cut_yu)
                        iZ = int(x + cut_xl)
                        self.data3d[iX][iY][iZ] = 1

    def get_output(self):
        return self.data3d

    def save(self, outputfile, filetype='pklz'):
        data = {
            'data3d': self.data3d,
            'voxelsize_mm': self.voxelsize_mm
        }

        misc.obj_to_file(data, outputfile, filetype=filetype)
        print "saved"
        #dw = datawriter.DataWriter()
        #dw.Write3DData(self.data3d, outputfile, filetype)

    def show(self):
        pyed = se.sed3(self.data3d)
        pyed.show()


