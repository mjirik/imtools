#! /usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/sed3"))

import numpy as np

import logging
logger = logging.getLogger(__name__)

import subprocess
import scipy
import scipy.ndimage

from .image_manipulation import *

def getVersionString():
    """
    Function return string with version information.
    It is performed by use one of three procedures: git describe,
    file in .git dir and file __VERSION__.
    """
    version_string = None
    try:
        version_string = subprocess.check_output(['git', 'describe'])
    except:
        logger.warning('Command "git describe" is not working')

    if version_string == None:  # noqa
        try:
            path_to_version = os.path.join(path_to_script,
                                           '../.git/refs/heads/master')
            with file(path_to_version) as f:
                version_string = f.read()
        except:
            logger.warning('Problem with reading file ".git/refs/heads/master"')

    if version_string == None:  # noqa
        try:
            path_to_version = os.path.join(path_to_script, '../__VERSION__')
            with file(path_to_version) as f:
                version_string = f.read()
            path_to_version = path_to_version + \
                              '  version number is created manually'

        except:
            logger.warning('Problem with reading file "__VERSION__"')

    return version_string


