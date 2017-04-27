#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import logging

logger = logging.getLogger(__name__)

import unittest
import numpy as np
import teigen.geometry3d as g3
import teigen.dili as dili


class DictListTestCase(unittest.TestCase):
    def test_ditc_flatten(self):
        data = {
            'a':1,
            'b':2,
            'c':{
                'aa':11,
                'bb':22,
                'cc':{
                    'aaa':111
                }
            }
        }
        dct = dili.flattenDict(data)
        dct = dict(dct)
        self.assertIn("cccaaa", dct.keys())


def main():
    unittest.main()
