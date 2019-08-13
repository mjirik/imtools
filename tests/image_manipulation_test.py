#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import logging
logger = logging.getLogger(__name__)
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
import unittest


import numpy as np

from imtools import image_manipulation as imm
from imtools import image_manipulation as imma
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
        crinfo_auto1 = imtools.image_manipulation.crinfo_from_specific_data(segmentation, [5])
        crinfo_auto2 = imtools.image_manipulation.crinfo_from_specific_data(segmentation, 5)
        crinfo_auto3 = imtools.image_manipulation.crinfo_from_specific_data(segmentation, [5,5, 5])

        crinfo_expected = [[0, 99], [20, 99], [45, 99]]

        self.assertEqual(crinfo_auto1, crinfo_expected)
        self.assertEqual(crinfo_auto1, crinfo_auto2)
        self.assertEqual(crinfo_auto1, crinfo_auto3)

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
        self.assertEqual(img_in.shape, img_uncropped.shape)

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


        self.assertEqual(img_in.shape, img_uncropped.shape)
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
            self.assertEqual(sonda_intensity_in, sonda_intensity_uncropped)

    def test_resize_to_shape(self):

        data = np.random.rand(3, 4, 5)
        new_shape = [5, 6, 6]
        data_out = imm.resize_to_shape(data, new_shape)
        # self.assertCountEqual(new_shape, data_out.shape)
        self.assertEqual(new_shape[0], data_out.shape[0])
        self.assertEqual(new_shape[1], data_out.shape[1])
        self.assertEqual(new_shape[2], data_out.shape[2])

    def test_resize_to_shape_no_new_unique_values(self):
        data = np.zeros([10, 15, 12])
        value1 = 1
        value2 = 2
        data[:5, :7, :6] = value1
        data[-5:, :7, :6] = value2

        expected_shape = [15, 15, 15]
        resized = imm.resize_to_shape(data, expected_shape)
        unique = np.unique(resized)

        self.assertEqual(resized.shape[0], expected_shape[0])
        self.assertEqual(resized.shape[1], expected_shape[1])
        self.assertEqual(resized.shape[2], expected_shape[2])
        self.assertEqual(resized[1, 1, 1], value1)
        self.assertEqual(resized[-2, 1, 1], value2)
        self.assertEqual(len(unique), 3)
        self.assertEqual(unique[0], 0)
        self.assertEqual(unique[1], 1)
        self.assertEqual(unique[2], 2)

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
        self.assertEqual(expected_shape[0], data_out.shape[0])
        self.assertEqual(expected_shape[1], data_out.shape[1])
        self.assertEqual(expected_shape[2], data_out.shape[2])
        # self.assertCountEqual(expected_shape, data_out.shape)

    def test_simple_get_nlabel(self):
        slab={"liver": 1, "porta": 2}
        val = imm.get_nlabel(slab, 2)
        self.assertEqual(val, 2)
        self.assertEqual(len(slab), 2)

    def test_simple_string_get_nlabel(self):
        slab={"liver": 1, "porta": 2}
        val = imm.get_nlabel(slab, "porta")
        self.assertEqual(val, 2)
        self.assertEqual(len(slab), 2)

    def test_simple_new_numeric_get_nlabel(self):
        slab={"liver": 1, "porta": 2}
        val = imm.get_nlabel(slab, 7)
        self.assertNotEqual(val, 1)
        self.assertNotEqual(val, 2)
        self.assertEqual(val, 7)

    def test_simple_new_string_get_nlabel(self):
        slab={"liver": 1, "porta": 2}
        val = imm.get_nlabel(slab, "cava")
        self.assertNotEqual(val, 1)
        self.assertNotEqual(val, 2)

    def test_simple_string_get_nlabel_return_string(self):
        slab={"liver": 1, "porta": 2}
        val = imm.get_nlabel(slab, "porta", return_mode="str")
        self.assertEqual(val, "porta")

    def test_simple_numeric_get_nlabel_return_string(self):
        slab={"liver": 1, "porta": 2}
        val = imm.get_nlabel(slab, 2, return_mode="str")
        self.assertEqual(val, "porta")

    def test_get_nlabels_single_label(self):
        slab={"liver": 1, "kindey": 15, "none":0}
        labels = 1
        val = imm.get_nlabels(slab, labels)
        self.assertEqual(val, 1)

    def test_get_nlabels_multiple(self):
        slab={"liver": 1, "porta": 2}
        val = imm.get_nlabels(slab, [2, "porta", "new", 7], return_mode="str")
        self.assertEqual(val[0], "porta")
        self.assertEqual(val[1], "porta")
        self.assertEqual(val[2], "3")
        self.assertEqual(val[3], "7")

    def test_get_nlabels_single(self):
        slab={"liver": 1, "porta": 2}

        val = imm.get_nlabels(slab, "porta", return_mode="int")
        self.assertEqual(val, 2)

    def test_get_nlabels_single_both(self):
        slab={"liver": 1, "porta": 2}

        val = imm.get_nlabels(slab, "porta", return_mode="both")
        self.assertEqual(val[0], 2)
        self.assertEqual(val[1], "porta")

    def test_select_objects_by_seeds(self):
        shape = [12, 15, 12]
        data = np.zeros(shape)
        value1 = 1
        value2 = 1
        data[:5, :7, :6] = value1
        data[-5:, :7, :6] = value2

        seeds = np.zeros(shape)
        seeds[9, 3:6, 3] = 1

        selected = imma.select_objects_by_seeds(data, seeds)
        # import sed3
        # ed =sed3.sed3(selected, contour=data, seeds=seeds)
        # ed.show()
        unique = np.unique(selected)
        #
        self.assertEqual(selected.shape[0], shape[0])
        self.assertEqual(selected.shape[1], shape[1])
        self.assertEqual(selected.shape[2], shape[2])
        self.assertEqual(selected[1, 1, 1], 0)
        self.assertEqual(selected[-2, 1, 1], 1)
        self.assertEqual(len(unique), 2)
        self.assertGreater(np.count_nonzero(data), np.count_nonzero(selected))

    def test_rotate(self):

        datap = imtools.sample_data.generate()
        data3d = datap["data3d"]
        phi_deg, theta_deg = imtools.image_manipulation.random_rotate_paramteres()
        data3d_rot = imtools.image_manipulation.rotate(data3d, phi_deg, theta_deg)
        # import sed3
        # ed = sed3.sed3(data3d_rot)
        # ed.show()
        # self.assertEqual(np.min(data3d), np.min(data3d_rot))
        # self.assertEqual(np.max(data3d), np.max(data3d_rot))

    def test_multiple_crop_and_uncrop_nearest_outside(self):
        """
        test combination of multiple crop
        """

        shape = [10, 11, 5]
        img_in = np.random.random(shape)

        crinfo1 = [[2, 8], [3, 9], [2, 5]]
        # crinfo2 = [[2, 5], [1, 5], [1, 2]]

        img_cropped = imma.crop(img_in, crinfo1)
        # img_cropped = imma.crop(img_cropped, crinfo2)

        # crinfo_combined = imma.combinecrinfo(crinfo1, crinfo2)

        img_uncropped = imma.uncrop(img_cropped, crinfo1, shape, outside_mode="nearest")

        # import sed3
        # ed = sed3.sed3(img_uncropped)
        # ed.show()
        self.assertTrue(img_uncropped[4, 4, 3] == img_in[4, 4, 3])

        self.assertTrue(img_uncropped[crinfo1[0][0], 5, 3] == img_uncropped[0, 5, 3], msg="pixels under crop")
        self.assertTrue(img_uncropped[5, crinfo1[1][0], 3] == img_uncropped[5, 0, 3], msg="pixels under crop")
        self.assertTrue(img_uncropped[7, 3, crinfo1[2][0]] == img_uncropped[7, 3, 0], msg="pixels under crop")

        self.assertTrue(img_uncropped[crinfo1[0][1] - 1, 5, 3] == img_uncropped[-1, 5, 3], msg="pixels over crop")
        self.assertTrue(img_uncropped[5, crinfo1[1][1] - 1, 3] == img_uncropped[5, -1, 3], msg="pixels over crop")
        self.assertTrue(img_uncropped[7, 3, crinfo1[2][1] - 1] == img_uncropped[7, 3, -1], msg="pixels over crop")

        # self.assertTrue(img_uncropped[crinfo1[0][1], 5 , 3] == img_uncropped[0, 5, 3], msg="pixels over crop")
        # self.assertTrue(img_uncropped[crinfo1[1][1], 5 , 3] == img_uncropped[1, 5, 3], msg="pixels over crop")
        # self.assertTrue(img_uncropped[crinfo1[2][1], 5 , 3] == img_uncropped[2, 5, 3], msg="pixels over crop")
        self.assertEqual(img_in.shape, img_uncropped.shape)

    def test_uncrop_with_none_crinfo(self):
        shape = [10, 10, 5]
        orig_shape = [15, 13, 7]
        img_in = np.random.random(shape)

        img_uncropped = imma.uncrop(img_in, crinfo=None, orig_shape=orig_shape)

        self.assertTrue(img_uncropped[-1, -1, -1] == 0)
        self.assertTrue(img_uncropped[4, 4, 3] == img_in[4, 4, 3])


    def test_uncrop_with_start_point_crinfo(self):
        shape = [10, 10, 5]
        orig_shape = [15, 13, 7]
        img_in = np.random.random(shape)
        crinfo = [5, 2, 1]

        img_uncropped = imma.uncrop(img_in, crinfo=crinfo, orig_shape=orig_shape)

        self.assertTrue(img_uncropped[-1, -1, -1] == 0)
        self.assertTrue(img_uncropped[4 + 5, 4 + 2, 3 + 1] == img_in[4 , 4, 3])

if __name__ == "__main__":
    unittest.main()
