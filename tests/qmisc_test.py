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


from imtools import qmisc
from imtools import misc
import imtools.sample_data


#

class QmiscTest(unittest.TestCase):
    interactivetTest = False
    # interactivetTest = True

    # @unittest.skip("waiting for implementation")
    def test_suggest_filename(self):
        """
        Testing some files. Not testing recursion in filenames. It is situation
        if there exist file0, file1, file2 and input file is file
        """
        filename = "mujsoubor"
        # import ipdb; ipdb.set_trace() # BREAKPOINT
        new_filename = misc.suggest_filename(filename, exists=True)
        self.assertTrue(new_filename == "mujsoubor2")

        filename = "mujsoubor112"
        new_filename = misc.suggest_filename(filename, exists=True)
        self.assertTrue(new_filename == "mujsoubor113")

        filename = "mujsoubor-2.txt"
        new_filename = misc.suggest_filename(filename, exists=True)
        self.assertTrue(new_filename == "mujsoubor-3.txt")

        filename = "mujsoubor-a24.txt"
        new_filename = misc.suggest_filename(filename, exists=False)
        self.assertTrue(new_filename == "mujsoubor-a24.txt")

    @unittest.skip("getVersionString is not used anymore")
    def test_getVersionString(self):
        """
        getVersionString is not used anymore
        """

        vfn = "../__VERSION__"
        existed = False
        if not os.path.exists(vfn):
            with open(vfn, 'a') as the_file:
                the_file.write('1.1.1\n')
            existed = False

        verstr = qmisc.getVersionString()

        self.assertTrue(type(verstr) == str)
        if existed:
            os.remove(vfn)


if __name__ == "__main__":
    unittest.main()
