#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module is provides funcions for dict lists and functions processing
"""
import logging
logger = logging.getLogger(__name__)
import collections
import inspect


def get_default_args(obj):
    if ("__init__" in dir(obj)):
        if inspect.isfunction(obj.__init__) or inspect.ismethod(obj.__init__):
            argspec = inspect.getargspec(obj.__init__)
        else:
            argspec = inspect.getargspec(obj)
    else:
        argspec = inspect.getargspec(obj)

    args = argspec.args[1:]
    defaults = argspec.defaults
    dc = collections.OrderedDict(zip(args, defaults))
    return dc

def subdict(dct, keys):
    if type(dct) == collections.OrderedDict:
        p = collections.OrderedDict()
    else:
        p = {}
    for key, value in dct.items():
        if key in keys:
            p[key] = value
    # p = {key: value for key, value in dct.items() if key in keys}
    return p

def kick_from_dict(dct, keys):
    if type(dct) == collections.OrderedDict:
        p = collections.OrderedDict()
    else:
        p = {}
    for key, value in dct.items():
        if key not in keys:
            p[key] = value

    # p = {key: value for key, value in dct.items() if key not in keys}
    return p

def split_dict(dct, keys):
    """
    Split dict into two subdicts based on keys
    :param dct:
    :param keys:
    :return: dict_in, dict_out
    """
    if type(dct) == collections.OrderedDict:
        dict_in = collections.OrderedDict()
        dict_out = collections.OrderedDict()
    else:
        dict_in = {}
        dict_out = {}

    for key, value in dct.items:
        if key in keys:
            dict_in[key] = value
        else:
            dict_out[key] = value
    return dict_in, dict_out

def recursive_update(d, u):
    """
    Dict recursive update.

    Based on Alex Martelli code on stackoverflow
    http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth?answertab=votes#tab-top

    :param d:
    :param u:
    :return:
    """
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = recursive_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d

from collections import Mapping
from itertools import chain
from operator import add

_FLAG_FIRST = object()

def flattenDict(d, join=add, lift=lambda x:x):
    """


    Based on ninjagecko code on stackoveflow
    http://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys

    :param d: dict to flatten
    :param join: join operation. To join keys with '_' use join=lambda a,b:a+'_'+b
    :param lift:  to have all hierarchy keys in lise use lift=lambda x:(x,))
    :return:

    For all keys from above hierarchy in list use:
    dict( flattenDict(testData, lift=lambda x:(x,)) )

    For all keys from abve hierarchy separated by '_' use:
    dict( flattenDict(testData, join=lambda a,b:a+'_'+b) )
    """
    results = []
    def visit(subdict, results, partialKey):
        for k,v in subdict.items():
            newKey = lift(k) if partialKey==_FLAG_FIRST else join(partialKey,lift(k))
            if isinstance(v,Mapping):
                visit(v, results, newKey)
            else:
                results.append((newKey,v))
    visit(d, results, _FLAG_FIRST)
    return results