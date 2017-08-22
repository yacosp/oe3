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
from oe3.coag import estim


__all__ = ('Song')

log = logging.getLogger('opus')


#------------------------------------------------------------------------------
class Song(object):
  """a song, including meta and compositional data"""

  def __init__(self, meta_path=None):
    """set defaults, load data"""

    self.id        = utils.tstamp()
    self.name      = ''
    self.date      = utils.datestr()
    self.estim     = None
    self.comp      = None   # (subcomp_class_name, subcomp_version)
    self.data_path = None
    self.sfile     = None   # song .flac file path
    if meta_path is not None: self.load(meta_path)

  def save(self, meta_path):
    """save song to filesystem

    songs are stored as three files:
      xxx.ogg      song soundfile
      xxx.pyd.bz2  song metadata (python dict as text)
      xxx.tar      song auxiliary files
    auxiliary files depend on each subcomposer
    """

    # XXX right now all this is (part of?) Spectrofoto._wrap_song()
    # clean up meta
    # encode .wav song to .ogg
    # save aux files
    # self.save_meta(meta_path)
    pass

  def load(self, meta_path):
    """load song from filesystem"""
    # XXX unfinished, but enough for bin/oe3 --reward
    log.info("loading song from %s", meta_path)
    self.load_meta(meta_path)
    self.estim = estim.ImageCollEstim(
      'var/estim/{}.pyd.bz2'.format(self.estim_id))
    # load aux files

  def save_meta(self, path):
    """save metadata to filesystem"""
    log.info("saving song metadata to %s", path)
    #meta = [k for k in self.__dict__ if k not in (estim)]
    meta = dict((k, v) for k, v in self.__dict__.iteritems() if k in (
      'anal', 'choices', 'comp', 'dur', 'id', 'img_anomaly',
      'img_distances', 'img_idx', 'img_medians', 'img_reduxes', 'seed'
    ))
    meta['estim_id']   = self.estim.id
    meta['estim_name'] = self.estim.name
    meta['sfile'] = self.id + '.ogg'
    utils.save_bzdict(path, meta)

  def load_meta(self, path):
    """load metadata from filesystem"""
    log.info("loading song metadata from %s", path)
    self.__dict__ = utils.load_bzdict(path)


#------------------------------------------------------------------------------
if __name__ == '__main__':
  from oe3.utils import run_doctest
  run_doctest('opus_song')
