#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import logging
logger = logging.getLogger(__name__)
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
import unittest


import numpy as np
import os


from imtools import image_manipulation as imm
import imtools.sample_data


#

class ImageManipulationTest(unittest.TestCase):
    interactivetTest = False
    # interactivetTest = True

    def test_store_to_SparseMatrix_and_back(self):
        data = np.zeros([4, 4, 4])
        data = np.zeros([4, 4, 4])
        data[1, 0, 3] = 1
        data[2, 1, 2] = 1
        data[0, 1, 3] = 2
        data[1, 2, 0] = 1
        data[2, 1, 1] = 3

        dataSM = imm.SparseMatrix(data)

        data2 = dataSM.todense()
        self.assertTrue(np.all(data == data2))

    def test_crop_and_uncrop(self):
        shape = [10, 10, 5]
        img_in = np.random.random(shape)

        crinfo = [[2, 8], [3, 9], [2, 5]]

        img_cropped = imm.crop(img_in, crinfo)

        img_uncropped = imm.uncrop(img_cropped, crinfo, shape)

        self.assertTrue(img_uncropped[4, 4, 3] == img_in[4, 4, 3])


    def test_crop_from_specific_data(self):

        datap = imtools.sample_data.generate()
        data3d = datap["data3d"]
        segmentation = datap["segmentation"]
        crinfo_auto1 = imtools.qmisc.crinfo_from_specific_data(segmentation, [5])
        crinfo_auto2 = imtools.qmisc.crinfo_from_specific_data(segmentation, 5)
        crinfo_auto3 = imtools.qmisc.crinfo_from_specific_data(segmentation, [5,5, 5])

        crinfo_expected = [[0, 99], [20, 99], [45, 99]]

        self.assertEquals(crinfo_auto1, crinfo_expected)
        self.assertEquals(crinfo_auto1, crinfo_auto2)
        self.assertEquals(crinfo_auto1, crinfo_auto3)

    def test_multiple_crop_and_uncrop(self):
        """
        test combination of multiple crop
        """

        shape = [10, 10, 5]
        img_in = np.random.random(shape)

        crinfo1 = [[2, 8], [3, 9], [2, 5]]
        crinfo2 = [[2, 5], [1, 4], [1, 2]]

        img_cropped = imm.crop(img_in, crinfo1)
        img_cropped = imm.crop(img_cropped, crinfo2)

        crinfo_combined = imm.combinecrinfo(crinfo1, crinfo2)

        img_uncropped = imm.uncrop(img_cropped, crinfo_combined, shape)

        self.assertTrue(img_uncropped[4, 4, 3] == img_in[4, 4, 3])
        self.assertEquals(img_in.shape, img_uncropped.shape)

    @unittest.skip("crinfo_combine should be tested in different way")
    def test_random_multiple_crop_and_uncrop(self):
        """
        test combination of multiple crop
        """

        shape = np.random.randint(10, 30, 3)
        # shape = [10, 10, 5]
        img_in = np.random.random(shape)

        crinfo1 = [
            sorted(np.random.randint(0, shape[0], 2)),
            sorted(np.random.randint(0, shape[1], 2)),
            sorted(np.random.randint(0, shape[2], 2))
        ]
        crinfo2 = [
            sorted(np.random.randint(0, shape[0], 2)),
            sorted(np.random.randint(0, shape[1], 2)),
            sorted(np.random.randint(0, shape[2], 2))
        ]

        img_cropped = imm.crop(img_in, crinfo1)
        img_cropped = imm.crop(img_cropped, crinfo2)

        crinfo_combined = imm.combinecrinfo(crinfo1, crinfo2)

        img_uncropped = imm.uncrop(img_cropped, crinfo_combined, shape)
        logger.debug("shape " + str(shape))
        logger.debug("crinfo_combined " + str(crinfo_combined))
        logger.debug("img_cropped.shape" + str(img_cropped.shape))
        logger.debug("img_uncropped.shape" + str(img_uncropped.shape))


        self.assertEquals(img_in.shape, img_uncropped.shape)
        # sonda indexes inside cropped area
        # cr_com = np.asarray(crinfo_combined)
        # if np.all((cr_com[:, 1] - cr_com[:, 0]) > 1):
        if np.all(img_cropped.shape > 1):
            # sometimes the combination of crinfo has zero size in one dimension
            sonda = np.array([
                np.random.randint(crinfo_combined[0][0], crinfo_combined[0][1] - 1),
                np.random.randint(crinfo_combined[1][0], crinfo_combined[1][1] - 1),
                np.random.randint(crinfo_combined[2][0], crinfo_combined[2][1] - 1),
                ])
            sonda_intensity_uncropped = img_uncropped[sonda[0], sonda[1], sonda[2]]
            sonda_intensity_in = img_in[sonda[0], sonda[1], sonda[2]]
            self.assertEquals(sonda_intensity_in, sonda_intensity_uncropped)

    def test_resize_to_shape(self):

        data = np.random.rand(3, 4, 5)
        new_shape = [5, 6, 6]
        data_out = imm.resize_to_shape(data, new_shape)
        # print data_out.shape
        # print data
        # print data_out
        self.assertItemsEqual(new_shape, data_out.shape)

    def test_fix_crinfo(self):
        crinfo = [[10, 15], [30, 40], [1, 50]]
        cri_fixed = imm.fix_crinfo(crinfo)

        # print crinfo
        # print cri_fixed

        self.assertTrue(cri_fixed[1, 1] == 40)
        self.assertTrue(cri_fixed[2, 1] == 50)

    def test_resize_to_mm(self):

        data = np.random.rand(3, 4, 5)
        voxelsize_mm = [2, 3, 1]
        new_voxelsize_mm = [1, 3, 2]
        expected_shape = [6, 4, 3]
        data_out = imm.resize_to_mm(data, voxelsize_mm, new_voxelsize_mm)
        print(data_out.shape)
        # print data
        # print data_out
        self.assertItemsEqual(expected_shape, data_out.shape)

if __name__ == "__main__":
    unittest.main()