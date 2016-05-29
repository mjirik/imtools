#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © %YEAR% %USER% <%MAIL%>
#
# Distributed under terms of the %LICENSE% license.

"""
%HERE%
"""

import logging

logger = logging.getLogger(__name__)
import argparse
import misc
import datetime


import gen_vtk_tree

class TreeGenerator:

    def __init__(self, generator_class='volume', generator_params=None):
        """

        :param generator_class: class with function add_cylinder(p1pix, p2pix, rad_mm) and get_output()
        :param generator_params:
        """
        self.rawdata = None
        self.tree_data = None
        self.data3d = None
        self.voxelsize_mm = [1, 1, 1]
        self.shape = None
        self.use_lar = False
        self.tree_label = None

        if generator_class in ['vol', 'volume']:
            import gt_volume
            generator_class = gt_volume.VolumeTreeGenerator
        elif generator_class  in ['lar']:
            import gt_lar
            generator_class = gt_lar.GTLar
        elif generator_class  in ['vtk']:
            import gen_vtk_tree
            generator_class = gen_vtk_tree.VTKTreeGenerator
        elif generator_class  in ['kunes']:
            import gt_lar_kunes
            generator_class = gt_lar_kunes.GTLar
        elif generator_class  in ['larsm']:
            import gt_lar_smooth
            generator_class = gt_lar_smooth.GTLarSmooth
        elif generator_class  in ['lar_nojoints']:
            import gt_lar
            generator_class = gt_lar.GTLar
            generator_params = {
                'endDistMultiplicator': 0,
                'use_joints': False
            }
        self.generator_class = generator_class
        self.generator_params = generator_params

    def fix_tree_structure(self, tree_raw_data):
        """
        Fix backward compatibility
        :param tree_raw_data:
        :return: fixed tree_raw_data
        """
        if 'graph' in tree_raw_data:
            gr = tree_raw_data.pop('graph')
            tree_raw_data['Graph'] = gr #{'tree1':gr}

        # if all keys in Graph a
        if all([type(k) != str for k in tree_raw_data['Graph'].keys()]):
            gr = tree_raw_data.pop('Graph')
            tree_raw_data['Graph'] = {'tree1':gr}


        # else:
        #     tree_raw_data = tree_raw_data['Graph']
        return tree_raw_data

    def importFromYaml(self, filename):
        rawdata = misc.obj_from_file(filename=filename, filetype='yaml')
        self.rawdata = self.fix_tree_structure(rawdata)

        tkeys = self.rawdata['Graph'].keys()
        if (self.tree_label is None) or (self.tree_label not in  tkeys):
            self.tree_label = tkeys[0]
        self.tree_data = self.rawdata['Graph'][self.tree_label]
        #
        # try:
        #     # key is usually "porta" or "microstructure"
        #     keys = self.rawdata['graph'].keys()
        #     self.tree_data = self.rawdata['graph'][keys[0]]
        # except:
        #     self.tree_data = self.rawdata['Graph']

    def generateTree(self):
        """
        | Funkce na vygenerování objemu stromu ze zadaných dat.
        | Generates output by defined generator. If VolumeTreeGenerator is used, output is data3d.
        """
        # LAR init
        if self.use_lar:
            import lar_vessels
            self.lv = lar_vessels.LarVessels()

        # use generator init
        if self.generator_params is None:
            self.generator = self.generator_class(self)
        else:
            self.generator = self.generator_class(self, **self.generator_params)

        for cyl_id in self.tree_data:
            logger.debug("CylinderId: " + str(cyl_id))
            cyl_data = self.tree_data[cyl_id]

            # try:
            #     cyl_data = self.data['graph']['porta'][cyl_id]
            # except:
            #     cyl_data = self.data['Graph'][cyl_id]

            # prvni a koncovy bod, v mm + radius v mm
            try:
                p1m = cyl_data['nodeA_ZYX_mm']  # souradnice ulozeny [Z,Y,X]
                p2m = cyl_data['nodeB_ZYX_mm']
                rad = cyl_data['radius_mm']
                self.generator.add_cylinder(p1m, p2m, rad, cyl_id)
            except Exception, e:
                # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

                logger.error(
                    "Segment id " + str(cyl_id) + ": error reading data from yaml!: "+str(e))
                # return

                # if self.use_lar:
                #     self.generator.add_cylinder(p1m, p2m, rad, in)
        logger.debug("cylinders generated")

        try:
            # generator could have finish() function
            self.generator.finish()
            logger.debug("joints generated")
        except:
            import traceback
            logger.debug(traceback.format_exc())

        output = self.generator.get_output()

        logger.debug("before visualization - generateTree()")
        if self.use_lar:
            self.lv.show()
        return output

    def generateTree_vtk(self):
        import vtk
        from vtk.util import numpy_support
        """
        | Funkce na vygenerování objemu stromu ze zadaných dat.
        | Veze pro generování pomocí VTK
        | !!! funguje špatně -> vstupní data musí být pouze povrchové body, jinak generuje ve výstupních datech dutiny

        """
        # get vtkPolyData
        tree_data = gen_vtk_tree.compatibility_processing(self.rawdata['Graph'])
        polyData = gen_vtk_tree.gen_tree(tree_data)

        polyData.GetBounds()
        # bounds = polyData.GetBounds()

        white_image = vtk.vtkImageData()
        white_image.SetSpacing(self.voxelsize_mm)
        white_image.SetDimensions(self.shape)
        white_image.SetExtent(
            [0, self.shape[0] - 1, 0, self.shape[1] - 1, 0, self.shape[2] - 1])
        # origin = [(bounds[0] + self.shape[0])/2, (bounds[1] + self.shape[1])/2, (bounds[2] + self.shape[2])/2]
        # white_image.SetOrigin(origin) #neni potreba?
        # white_image.SetScalarTypeToUnsignedChar()
        white_image.AllocateScalars()

        # fill the image with foreground voxels: (still black until stecil)
        inval = 255
        outval = 0
        count = white_image.GetNumberOfPoints()
        for i in range(0, count):
            white_image.GetPointData().GetScalars().SetTuple1(i, inval)

        pol2stencil = vtk.vtkPolyDataToImageStencil()
        pol2stencil.SetInput(polyData)

        # pol2stencil.SetOutputOrigin(origin) # TOHLE BLBNE
        pol2stencil.SetOutputSpacing(self.voxelsize_mm)
        pol2stencil.SetOutputWholeExtent(white_image.GetExtent())
        pol2stencil.Update()

        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInput(white_image)
        imgstenc.SetStencil(pol2stencil.GetOutput())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(outval)
        imgstenc.Update()

        # VTK -> Numpy
        vtk_img_data = imgstenc.GetOutput()
        vtk_data = vtk_img_data.GetPointData().GetScalars()
        numpy_data = numpy_support.vtk_to_numpy(vtk_data)
        numpy_data = numpy_data.reshape(
            self.shape[0], self.shape[1], self.shape[2])
        numpy_data = numpy_data.transpose(2, 1, 0)

        self.data3d = numpy_data

    def saveToFile(self, outputfile, filetype):
        self.generator.save(outputfile, filetype)

    def show(self):
        self.generator.show()


def main():
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # input parser
    parser = argparse.ArgumentParser(
        description='Histology analyser reporter. Try: \
python src/gt_volume.py -i ./tests/hist_stats_test.yaml'
    )
    parser.add_argument(
        '-i', '--inputfile',
        default=None,
        required=True,
        help='input file, yaml file'
    )
    parser.add_argument(
        '-o', '--outputfile',
        default=None,
        help='output file, .raw, .dcm, .tiff, given by extension '
    )
    parser.add_argument(
        '-ot', '--outputfiletype',
        default='pkl',
        help='output file type.  raw, dcm, tiff, or pkl,   default is pkl, '
    )
    parser.add_argument(
        '-vs', '--voxelsize',
        default=[1.0, 1.0, 1.0],
        type=float,
        metavar='N',
        nargs='+',
        help='size of voxel (ZYX)'
    )
    parser.add_argument(
        '-ds', '--datashape',
        default=[200, 200, 200],
        type=int,
        metavar='N',
        nargs='+',
        help='size of output data in pixels for each axis (ZYX)'
    )
    parser.add_argument(
        '-g', '--generator',
        default='vol',
        type=str,
        help='Volume or surface model can be generated by use this option. \
                Use "vol", "volume" for volumetric model. For LAR surface model\
                use "lar". For VTK file use "vtk".'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    parser.add_argument(
        '-l', '--useLar', action='store_true',
        help='Use LAR')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    startTime = datetime.now()

    generator_params = None
    generator_class = args.generator

    # if args.generator == "vtk":
    #     import gen_vtk_tree
    #     gen_vtk_tree.vt2vtk_file(args.inputfile, args.outputfile)
    #     return

    tg = TreeGenerator(generator_class, generator_params)
    tg.importFromYaml(args.inputfile)
    tg.voxelsize_mm = args.voxelsize
    tg.shape = args.datashape
    tg.use_lar = args.useLar
    data3d = tg.generateTree()

    logger.info("TimeUsed:" + str(datetime.now() - startTime))
    # volume_px = sum(sum(sum(data3d)))
    # volume_mm3 = volume_px * \
    #     (tg.voxelsize_mm[0] * tg.voxelsize_mm[1] * tg.voxelsize_mm[2])
    # logger.info("Volume px:" + str(volume_px))
    # logger.info("Volume mm3:" + str(volume_mm3))

    # vizualizace
    logger.debug("before visualization")
    tg.show()
    logger.debug("after visualization")

    # ukládání do souboru
    if args.outputfile is not None:
        tg.saveToFile(args.outputfile, args.outputfiletype)

if __name__ == "__main__":
    main()
