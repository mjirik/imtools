#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
import unittest


import numpy as np
import os


from imtools import image_manipulation as imma
from imtools import misc
import imtools.sample_data


#

class QmiscTest(unittest.TestCase):
    interactivetTest = False
    # interactivetTest = True

    # @unittest.skip("waiting for implementation")
    def test_get_nlabels(self):
        slab = {"liver":1, "porta":2, "none":0}
        self.assertEqual(imma.get_nlabels(slab, "porta"), 2)
        self.assertEqual(imma.get_nlabels(slab, 1), 1)
        # after prev line can be number one added twice {"liver":1, "1":1}
        self.assertFalse("1" in slab.keys())
        self.assertEqual(imma.get_nlabels(slab, ["porta", "liver"]), [2, 1])

if __name__ == "__main__":
    unittest.main()
