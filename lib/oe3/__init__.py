# oe3 lib/oe3/__init__.py
#
# oveja electrica - oe3.0.1:bleu
# copyright (c) 2003-2017 santiago pereson <yaco@yaco.net>
# http://yaco.net/oe/
#
# this file is part of 'oe3', which is released under the gnu general public
# license, version 3. see LICENSE for full license details.
#


"""a composer of non-human music"""


from __future__ import division

import logging
import logging.config
import os
import time

import arrow

from .utils import save_json, shexec, load_json, compress_logs


__author__    = "santiago pereson <yaco@yaco.net>"
__copyright__ = "copyright (c) 2003-2017 santiago pereson <yaco@yaco.net>"
__license__   = "GPL v3"
__version__   = '3.1'

__all__ = ['vagab', 'coag', 'gatil', 'comp', 'opus', 'interp', 'crit',
           'utils', 'oe3_path', 'anatta']

oe3_path = os.path.abspath(os.path.dirname(__file__) + '/../..')

# load general oe3 state
if not os.path.exists(os.path.join(oe3_path, 'var/lib/oe3.json')):
  # init anatta
  # XXX this should be @ anatta.py
  anatta = {'name':      'unknown',
            'born_date': arrow.get().format('YYYY-MM-DD HH:mm:ss ZZ'),
            'born_sys':  shexec('uname -a')
  }
  save_json('var/lib/oe3.json', {'anatta': anatta})
else:
  anatta = load_json('var/lib/anatta.json')

# set up logging and compress old logs
wd = os.getcwd()
os.chdir(oe3_path)
logging.config.fileConfig('etc/logging.conf')
logging.Formatter.converter = time.gmtime
compress_logs()
os.chdir(wd)

# cleanup
del division, logging, os, time, arrow, \
  save_json, shexec, load_json, compress_logs, wd
