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


__author__    = "santiago pereson <yaco@yaco.net>"
__copyright__ = "copyright (c) 2003-2017 santiago pereson <yaco@yaco.net>"
__license__   = "GPL v3"

__version__ = '3.0.1'


#------------------------------------------------------------------------------

# path
oe3_path = os.path.abspath(os.path.dirname(__file__) + '/../..')

# set up logging
wd = os.getcwd()
os.chdir(oe3_path)
logging.config.fileConfig('etc/logging.conf')
os.chdir(wd)
del wd
