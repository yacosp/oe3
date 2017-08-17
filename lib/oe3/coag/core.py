# oe3 lib/oe3/coag/core.py
#
# oveja electrica - oe3.0.1:bleu
# copyright (c) 2003-2017 santiago pereson <yaco@yaco.net>
# http://yaco.net/oe/
#
# this file is part of 'oe3', which is released under the gnu general public
# license, version 3. see LICENSE for full license details.
#

"""coag subsystem core"""


from __future__ import division

import logging
import os
import random

from oe3 import oe3_path, utils
from oe3.coag import estim


__version__ = '0.0.1'

conf_path = os.path.join(oe3_path, 'etc/coag.conf')
log = logging.getLogger('coag')


#------------------------------------------------------------------------------
class Coag(object):
  """a stimuli database"""

  def __init__(self):
    """load conf, load main index, generate other indexes"""
    self.conf  = utils.load_dict(conf_path)
    try:
      self.index = utils.load_bzdict(
        os.path.join(oe3_path, self.conf['index_file']))
    except Exception, e:
      log.critical("couldn't load estim index!")
      raise
    self._subindex = {}
    self._load_state(self.conf['state_path'])

  def __del__(self):
    """save state"""
    self._save_state(self.conf['state_path'])

  def subindex(self, key, val):
    """get a subindex for the specified key and value"""
    if key not in self._subindex:
      self._subindex[key] = {}
    if val not in self._subindex[key]:
      self._subindex[key][val] = self._gen_subindex(key, val)
    return self._subindex[key][val]

  def rand_estim(self, key, val):
    """get a random estim where <key> == <val>"""
    meta  = random.choice(self.subindex(key, val))
    etype = self.conf['type_map'][meta['type']]
    return getattr(estim, etype)(meta['meta_path'])

  def _gen_subindex(self, key, val):
    """generate a subindex"""
    log.info("generating subindex for %s: %s", key, val)
    return [item for item in self.index if item[key] == val]

  def _save_state(self, path):
    pass

  def _load_state(self, path):
    pass

#------------------------------------------------------------------------------
if __name__ == '__main__':
  from oe3.utils import run_doctest
  run_doctest('coag_core')
