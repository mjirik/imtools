#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import os
import os.path

from nose.plugins.attrib import attr
path_to_script = os.path.dirname(os.path.abspath(__file__))
import unittest

import shutil
import numpy as np


# from imtools import qmisc
# from imtools import misc


import imtools.sample_data as sd
#

class SampleDataTest(unittest.TestCase):
    interactivetTest = False
    # interactivetTest = True
    def sample_data_test(self):
        sd.get_sample_data("head", "delete_head")
        self.assertTrue(os.path.exists("./delete_head/matlab/examples/sample_data/DICOM/digest_article/brain_001.dcm"))
        shutil.rmtree("delete_head")


        # import imtools.vesseltree_export as vt
        # yaml_input = os.path.join(path_to_script, "vt_biodur.yaml")
        # yaml_output = os.path.join(path_to_script, "delme_esofspy.txt")
        # vt.vt2esofspy(yaml_input, yaml_output)

    def sample_data_batch_test(self):
        sd.get_sample_data(["head", "exp_small"], "delete_sample_data")
        self.assertTrue(os.path.exists("./delete_sample_data/exp_small/seeds/org-liver-orig003-seeds.pklz"))
        self.assertTrue(os.path.exists("./delete_sample_data/matlab/examples/sample_data/DICOM/digest_article/brain_001.dcm"))
        shutil.rmtree("delete_sample_data")

if __name__ == "__main__":
    unittest.main()
