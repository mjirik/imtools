#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import logging

logger = logging.getLogger(__name__)

import unittest
import imtools.dili as dili


class DictListTestCase(unittest.TestCase):
    def test_ditc_flatten(self):
        data = {
            'a': 1,
            'b': 2,
            'c': {
                'aa': 11,
                'bb': 22,
                'cc': {
                    'aaa': 111
                }
            }
        }
        dct = dili.flatten_dict(data)
        dct = dict(dct)
        self.assertIn("cccaaa", dct.keys())

    def test_ditc_flatten_with_separator(self):
        data = {
            'a': 1,
            'b': 2,
            'c': {
                'aa': 11,
                'bb': 22,
                'cc': {
                    'aaa': 111
                }
            }
        }
        dct = dili.flatten_dict(data, separator=";")
        dct = dict(dct)
        self.assertIn("c;cc;aaa", dct.keys())

    def test_list_filter(self):
        lst = ["aa", "sss", "aaron", "rew"]
        output = dili.list_filter(lst, notstartswith="aa")
        self.assertTrue(["sss", "rew"] == output)


def main():
    unittest.main()
