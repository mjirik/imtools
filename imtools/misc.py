#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

# import sys

logger = logging.getLogger(__name__)

from io3d.misc import obj_from_file, obj_to_file, suggest_filename

# path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "./extern/sPickle"))
# import numpy as np

# try:
#     import cPickle as pickle
# except ImportError:
#     import pickle

# def suggest_filename(file_path, exists=None):
#     """
#     Append a string with number to its filename if the path exist.
#     :param file_path: input file path
#     :param exists: You can force path exists check. It is usefull for debug issues.
#     """
#     import os.path
#     import re
#     if not isinstance(exists, bool):
#         exists = os.path.exists(file_path)
#
#     if exists:
#         file_path, file_extension = os.path.splitext(file_path)
#         # print(file_path)
#         m = re.search(r"\d+$", file_path)
#         if m is None:
#             # cislo = 2
#             new_cislo_str = "2"
#         else:
#             cislostr = (m.group())
#             cislo = int(cislostr) + 1
#             file_path = file_path[:-len(cislostr)]
#             new_cislo_str = str(cislo)
#
#         file_path = file_path + new_cislo_str + file_extension  # .zfill(2)
#         # trorcha rekurze
#         file_path = suggest_filename(file_path)
#
#     return file_path
#

# def obj_from_file(filename='annotation.yaml', filetype='yaml'):
#     ''' Read object from file '''
#
#     if filetype == 'auto':
#         _, ext = os.path.splitext(filename)
#         filetype = ext[1:]
#
#     if filetype == 'yaml':
#         import yaml
#         f = open(filename, 'rb')
#         obj = yaml.load(f)
#         f.close()
#     elif filetype in ('pickle', 'pkl', 'pklz', 'picklezip'):
#         fcontent = read_pkl_and_pklz(filename)
#         obj = pickle.loads(fcontent)
#     else:
#         logger.error('Unknown filetype ' + filetype)
#     return obj
#
#
# def read_pkl_and_pklz(filename):
#     """
#     Try read zipped or not zipped pickle file
#     """
#     fcontent = None
#     try:
#         import gzip
#         f = gzip.open(filename, 'rb')
#         fcontent = f.read()
#         f.close()
#     except IOError as e:
#         # if the problem is in not gzip file
#         logger.warning("Input gzip exception: " + str(e))
#         f = open(filename, 'rb')
#         fcontent = f.read()
#         f.close()
#     except Exception as e:
#         # other problem
#         import traceback
#         logger.error("Input gzip exception: " + str(e))
#         logger.error(traceback.format_exc())
#
#     return fcontent
#
#
# def obj_to_file(obj, filename='annotation.yaml', filetype='yaml'):
#     '''Writes annotation in file.
#
#     Filetypes:
#         yaml
#         pkl, pickle
#         pklz, picklezip
#     '''
#     # import json
#     # with open(filename, mode='w') as f:
#     #    json.dump(annotation,f)
#
#     # write to yaml
#     d = os.path.dirname(os.path.abspath(filename))
#     if not os.path.exists(d):
#         os.makedirs(d)
#
#     if filetype == 'auto':
#         _, ext = os.path.splitext(filename)
#         filetype = ext[1:]
#
#     if filetype == 'yaml':
#         f = open(filename, 'wb')
#         import yaml
#         yaml.dump(obj, f)
#         f.close
#     elif filetype in ('pickle', 'pkl'):
#         with open(filename, 'wb') as f:
#             pickle.dump(obj, f)
#
#     elif filetype in ('streamingpicklezip', 'spklz'):
#         # this is not working :-(
#         import gzip
#         import sPickle as spickle
#         f = gzip.open(filename, 'wb', compresslevel=1)
#         # f = open(filename, 'wb')
#         spickle.s_dump(obj, f)
#         f.close
#     elif filetype in ('picklezip', 'pklz'):
#         import gzip
#         with gzip.open(filename, 'wb', compresslevel=1) as f:
#             # f = open(filename, 'wb')
#             pickle.dump(obj, f)
#     elif filetype in('mat'):
#
#         import scipy.io as sio
#         sio.savemat(filename, obj)
#     else:
#         logger.error('Unknown filetype ' + filetype)


