# oe3 lib/oe3/opus/song.py
#
# oveja electrica - oe3.0.1:bleu
# copyright (c) 2003-2017 santiago pereson <yaco@yaco.net>
# http://yaco.net/oe/
#
# this file is part of 'oe3', which is released under the gnu general public
# license, version 3. see LICENSE for full license details.
#

"""song class"""


from __future__ import division

import logging

from oe3 import utils


__all__ = ('Song')

log = logging.getLogger('opus')


#------------------------------------------------------------------------------
class Song(object):
  """a song, including meta and compositional data"""

  def __init__(self, estim, comp):
    """set default metadata"""
    # subcomposers will add more attributes to the instance
    self.id    = utils.tstamp()
    self.estim = estim   # song stimulus
    self.comp  = comp    # (subcomp_class_name, subcomp_version)
    self.sfile = None    # song .flac file path

  def save(self, flac_path, data_path, meta_path):
    """save song to filesystem

    songs are stored as three files:
      xxx.ogg      song soundfile
      xxx.pyd.bz2  song metadata (python dict as text)
      xxx.tar      song auxiliary files
    auxiliary files depend on each subcomposer
    """
    pass

  def load(self, meta_path):
    """load song from filesystem"""
    log.info("loading song from %s", meta_path)
    pass

  def _save_meta(self, path):
    """save metadata to filesystem"""
    meta = [k for k in self.__dict__ if k not in (estim)]

  def _load_meta(self, path):
    """load metadata from filesystem"""
    pass


#------------------------------------------------------------------------------
if __name__ == '__main__':
  from oe3.utils import run_doctest
  run_doctest('opus_song')
