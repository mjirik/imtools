#! /usr/bin/python
# -*- coding: utf-8 -*-


# import funkcí z jiného adresáře
import os.path
import sys
path_to_script = os.path.dirname(os.path.abspath(__file__))
pth = os.path.join(path_to_script, "../../seededitorqt/")
sys.path.insert(0, pth)

path_to_script = os.path.dirname(os.path.abspath(__file__))
import unittest

from nose.plugins.attrib import attr
import numpy as np

from PyQt4.QtGui import QApplication
import sed3

from imtools import segmentation as imsegmentation


class SegmentationTest(unittest.TestCase):
    interactiveTest = False

    # @unittest.skip("demonstrating skipping")
    # @unittest.skipIf(not interactiveTest, "interactive test")
    # @attr('interactive')

    def synthetic_data(self):
        # data
        slab = {'none': 0, 'liver': 1, 'porta': 2}
        voxelsize_mm = np.array([1.0, 1.0, 1.2])

        segm = np.zeros([256, 256, 80], dtype=np.int16)
        vessel_seeds=np.zeros_like(segm, dtype=np.int8)

        # liver
        segm[70:180, 40:190, 30:60] = slab['liver']
        # porta
        segm[120:130, 70:190, 40:45] = slab['porta']
        segm[80:130, 100:110, 40:45] = slab['porta']
        segm[120:170, 130:135, 40:44] = slab['porta']

        vessel_seeds[85:90, 105, 42] = 1

        data3d = np.zeros(segm.shape)
        data3d[segm == slab['liver']] = 156
        data3d[segm == slab['porta']] = 206
        noise = (np.random.normal(0, 30, segm.shape))  # .astype(np.int16)
        data3d = (data3d + noise).astype(np.int16)


        datap = {
            "data3d": data3d,
            "segmentation": segm,
            "slab": slab,
            "seeds": vessel_seeds,
            "voxelsize_mm": voxelsize_mm
        }
        return datap

    def test_synthetic_data_segmentation(self):
        """
        Function uses organ_segmentation  for synthetic box object
        segmentation.
        """
        datap = self.synthetic_data()
        data3d = datap["data3d"]
        segm = datap["segmentation"]
        voxelsize_mm = datap["voxelsize_mm"]
        slab = datap["slab"]

        outputTmp = imsegmentation.vesselSegmentation(data3d, segmentation=(segm == slab['liver']), threshold=180,
                                                      voxelsize_mm=voxelsize_mm, inputSigma=0.15,
                                                      aoi_dilation_iterations=2, nObj=1, biggestObjects=True,
                                                      interactivity=False, binaryClosingIterations=5,
                                                      binaryOpeningIterations=1)

        # @TODO opravit chybu v vesselSegmentation
        outputTmp = (outputTmp == 2)
        errim = np.abs(
            outputTmp.astype(np.int) - (segm == slab['porta']).astype(np.int)
        )

        # ověření výsledku
        # pyed = sed3.sed3(errim, contour=segm==slab['porta'])
        # pyed.show()
        # evaluation
        sum_of_wrong_voxels = np.sum(errim)
        sum_of_voxels = np.prod(segm.shape)
        errorrate = sum_of_wrong_voxels/sum_of_voxels

        self.assertLess(errorrate, 0.1)

    def test_synthetic_data_segmentation_with_vessel_seed(self):
        """
        Function uses organ_segmentation  for synthetic box object
        segmentation.
        """

        datap = self.synthetic_data()
        data3d = datap["data3d"]
        segm = datap["segmentation"]
        voxelsize_mm = datap["voxelsize_mm"]
        slab = datap["slab"]
        vessel_seeds = datap["seeds"]

        # pyed = sed3.sed3(data3d, seeds=vessel_seeds)
        # pyed.show()

        outputTmp = imsegmentation.vesselSegmentation(data3d, segmentation=segm, voxelsize_mm=voxelsize_mm,
                                                      inputSigma=0.15, aoi_dilation_iterations=2, nObj=1,
                                                      biggestObjects=True, seeds=vessel_seeds, interactivity=False,
                                                      binaryClosingIterations=5, binaryOpeningIterations=1,
                                                      aoi_label=["liver", "porta"], slab=slab)

        # @TODO opravit chybu v vesselSegmentation
        outputTmp = (outputTmp == 2)
        errim = np.abs(
            outputTmp.astype(np.int) - (segm == slab['porta']).astype(np.int)
        )

        # ověření výsledku
        # pyed = sed3.sed3(errim, contour=segm==slab['porta'])
        # pyed.show()
        # evaluation
        sum_of_wrong_voxels = np.sum(errim)
        sum_of_voxels = np.prod(segm.shape)
        errorrate = sum_of_wrong_voxels/sum_of_voxels

        self.assertLess(errorrate, 0.1)

    @attr('interactive')
    def test_synthetic_data_segmentation_interactive_check(self):
        """
        Function uses organ_segmentation  for synthetic box object
        segmentation.
        """

        datap = self.synthetic_data()
        data3d = datap["data3d"]
        segm = datap["segmentation"]
        voxelsize_mm = datap["voxelsize_mm"]
        slab = datap["slab"]
# @TODO je tam bug, prohlížeč neumí korektně pracovat s doubly
        import sys
        app = QApplication(sys.argv)
#        #pyed = QTSeedEditor(noise )
#        pyed = QTSeedEditor(data3d)
#        pyed.exec_()
#        #img3d = np.zeros([256,256,80], dtype=np.int16)

        # pyed = sed3.sed3(data3d)
        # pyed.show()

        outputTmp = imsegmentation.vesselSegmentation(data3d, segmentation=(segm == slab['liver']), threshold=180,
                                                      voxelsize_mm=voxelsize_mm, inputSigma=0.15,
                                                      aoi_dilation_iterations=2, nObj=1, biggestObjects=True,
                                                      interactivity=True,
                                                      binaryClosingIterations=5,
                                                      binaryOpeningIterations=1)

# ověření výsledku
        pyed = sed3.sed3(outputTmp, contour=segm==slab['porta'])
        pyed.show()

# @TODO opravit chybu v vesselSegmentation
        outputTmp = (outputTmp == 2)
        errim = np.abs(
            outputTmp.astype(np.int) - (segm == slab['porta']).astype(np.int)
        )

# ověření výsledku
        # pyed = sed3.sed3(errim, contour=segm==slab['porta'])
        # pyed.show()
# evaluation
        sum_of_wrong_voxels = np.sum(errim)
        sum_of_voxels = np.prod(segm.shape)

        # print "wrong ", sum_of_wrong_voxels
        # print "voxels", sum_of_voxels

        errorrate = sum_of_wrong_voxels/sum_of_voxels

        # import pdb; pdb.set_trace()

        self.assertLess(errorrate, 0.1)

    # @unittest.skipIf(os.environ.get("TRAVIS", True), "Skip on Travis-CI")
    def test_uiThreshold_qt(self):
        """
        UI threshold segmentation without binary close
        """
        from imtools import uiThreshold

        datap = self.synthetic_data()
        data3d = datap["data3d"]
        # segm = datap["segmentation"]
        voxelsize_mm = datap["voxelsize_mm"]
        # slab = datap["slab"]

        data3d[100:150, 58:70, 50:55] += 50
        app = QApplication(sys.argv)
        # pyed = sed3.sed3(data3d)
        # pyed.show()

        uiT = uiThreshold.uiThresholdQt(

            data3d,  # .astype(np.uint8),
            # segmentation=(segm == slab['liver']),  # .astype(np.uint8),
            # segmentation = oseg.orig_scale_segmentation,
            voxel=voxelsize_mm,
            threshold=180,
            inputSigma=0.15,
            nObj=1,
            interactivity=False,
            # interactivity=True,
            biggestObjects=True,
            # biggestObjects=False,
            binaryClosingIterations=0,
            binaryOpeningIterations=0)

        # outputTmp = uiT.run()

    @unittest.skipIf(os.environ.get("TRAVIS", True), "Skip on Travis-CI")
    def test_uiThreshold_binary_close_with_synthetic_data(self):
        """
        Function uses ui threshold for synthetic box object
        segmentation with binary close
        """
        # TODO check the result better
        from imtools import uiThreshold

        datap = self.synthetic_data()
        data3d = datap["data3d"]
        segm = datap["segmentation"]
        voxelsize_mm = datap["voxelsize_mm"]
        slab = datap["slab"]

        data3d[100:150, 58:70, 50:55] += 50
        # @TODO je tam bug, prohlížeč neumí korektně pracovat s doubly
        import sys
        app = QApplication(sys.argv)
        # pyed = sed3.sed3(data3d)
        # pyed.show()

        uiT = uiThreshold.uiThresholdQt(

            data3d,  # .astype(np.uint8),
            # segmentation=(segm == slab['liver']),  # .astype(np.uint8),
            # segmentation = oseg.orig_scale_segmentation,
            voxel=voxelsize_mm,
            threshold=180,
            inputSigma=0.15,
            nObj=1,
            interactivity=False,
            # interactivity=True,
            biggestObjects=True,
            # biggestObjects=False,
            binaryClosingIterations=5,
            binaryOpeningIterations=1)

        outputTmp = uiT.run()

        # ověření výsledku
        # pyed = sed3.sed3(outputTmp, contour=segm==slab['porta'])
        # pyed.show()

        outputTmp = (outputTmp == 2)
        errim = np.abs(
            outputTmp.astype(np.int) - (segm == slab['porta']).astype(np.int)
        )

        # evaluation
        sum_of_wrong_voxels = np.sum(errim)
        sum_of_voxels = np.prod(segm.shape)

        errorrate = sum_of_wrong_voxels/sum_of_voxels

        # import pdb; pdb.set_trace()
        self.assertLess(errorrate, 0.1)


if __name__ == "__main__":
    unittest.main()
