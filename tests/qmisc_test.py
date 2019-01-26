#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
import unittest


import numpy as np
import os


from imtools import qmisc
from imtools import misc


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
        # self.assertTrue(new_filename == "mujsoubor2")
        self.assertEqual(new_filename, "mujsoubor_2")

        filename = "mujsoubor_112"
        new_filename = misc.suggest_filename(filename, exists=True)
        self.assertTrue(new_filename == "mujsoubor_113")

        filename = "mujsoubor_2.txt"
        new_filename = misc.suggest_filename(filename, exists=True)
        self.assertTrue(new_filename == "mujsoubor_3.txt")

        filename = "mujsoubor27.txt"
        new_filename = misc.suggest_filename(filename, exists=True)
        self.assertTrue(new_filename == "mujsoubor27_2.txt")

        filename = "mujsoubor-a24.txt"
        new_filename = misc.suggest_filename(filename, exists=False)
        self.assertEqual(new_filename, "mujsoubor-a24.txt", "Rewrite")

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

    def test_obj_to_and_from_file_yaml(self):
        testdata = np.random.random([4, 4, 3])
        test_object = {'a': 1, 'data': testdata}

        filename = 'test_obj_to_and_from_file.yaml'
        misc.obj_to_file(test_object, filename, 'yaml')
        saved_object = misc.obj_from_file(filename, 'yaml')

        self.assertTrue(saved_object['a'] == 1)
        self.assertTrue(saved_object['data'][1, 1, 1] == testdata[1, 1, 1])

        os.remove(filename)

    def test_obj_to_and_from_file_pickle(self):
        testdata = np.random.random([4, 4, 3])
        test_object = {'a': 1, 'data': testdata}

        filename = 'test_obj_to_and_from_file.pkl'
        misc.obj_to_file(test_object, filename, 'pickle')
        saved_object = misc.obj_from_file(filename, 'pickle')

        self.assertTrue(saved_object['a'] == 1)
        self.assertTrue(saved_object['data'][1, 1, 1] == testdata[1, 1, 1])

        os.remove(filename)

    # def test_obj_to_and_from_file_exeption(self):
    #    test_object = [1]
    #    filename = 'test_obj_to_and_from_file_exeption'
    #    self.assertRaises(misc.obj_to_file(test_object, filename ,'yaml'))

    def test_obj_to_and_from_file_with_directories(self):
        import shutil
        testdata = np.random.random([4, 4, 3])
        test_object = {'a': 1, 'data': testdata}

        dirname = '__test_write_and_read'
        filename = '__test_write_and_read/test_obj_to_and_from_file.pkl'

        misc.obj_to_file(test_object, filename, 'pickle')
        saved_object = misc.obj_from_file(filename, 'pickle')

        self.assertTrue(saved_object['a'] == 1)
        self.assertTrue(saved_object['data'][1, 1, 1] == testdata[1, 1, 1])

        shutil.rmtree(dirname)


if __name__ == "__main__":
    unittest.main()
