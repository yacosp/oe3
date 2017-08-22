# oe3 lib/oe3/coag/estim.py
#
# oveja electrica - oe3.0.1:bleu
# copyright (c) 2003-2017 santiago pereson <yaco@yaco.net>
# http://yaco.net/oe/
#
# this file is part of 'oe3', which is released under the gnu general public
# license, version 3. see LICENSE for full license details.
#

"""stimuli classes"""


from __future__ import division

import logging
import os
import pprint
import tarfile
import tempfile

from PIL import Image

from oe3 import utils


log = logging.getLogger('coag')


#------------------------------------------------------------------------------
class Estim(object):
  """base stimulus"""

  def __init__(self, meta_path=None):
    """set defaults, load data"""
    self.id      = utils.tstamp()   # ie: 19711021.200000
    self.name    = ''               # ie: twexus_lux_7
    self.type    = 'base'           # ie: 'imagecoll'
    self.source  = None             # ie: 'http://www.twexus.com/
    self.date    = utils.datestr()  # date this estim was acquired
    self.used    = []               # XXX list of songs that used this estim
    self.data_path = None           # data file path (relative)
    if meta_path is not None: self.load(meta_path)

  def __len__(self):
    """estim size (definition depends on estim type)"""
    return 0

  def save(self, meta_path=None):
    """
    save estim to filesystem

    estims are stored as two files:
      xxx.pyd.bz2  metadata (bzip2-compressed python dict as text)
      xxx.xxx      estim data
    the estim data file type and content depends on the estim type
    """
    if meta_path is not None:
      self.save_meta(meta_path)
    # data should be saved to self.data_path

  def load(self, meta_path=None):
    """load estim from filesystem"""
    if meta_path is not None:
      self.load_meta(meta_path)
    # data should be loaded from self.data_path

  def save_meta(self, path):
    """save metadata to filesystem"""
    self.size = len(self)
    utils.save_bzdict(path, self.__dict__)
    del self.size

  def load_meta(self, path):
    """load metadata from filesystem"""
    self.__dict__ = utils.load_bzdict(path)

#------------------------------------------------------------------------------
class ImageCollEstim(Estim):
  """image collection stimulus"""

  def __init__(self, meta_path=None):
    """set defaults, load data"""
    Estim.__init__(self)
    self.type      = 'imagecoll'
    self.images    = []                   # image (pil object) list
    self.data_path = '%s.tar' % self.id
    self._index    = {}
    if meta_path is not None: self.load(meta_path)

  def __len__(self):
    """collection size"""
    return len(self.images)

  def save(self, meta_path=None):
    """save estim to filesystem"""
    log.info("saving estim to %s", meta_path)
    tmp = self.images[:]
    index = self._index; del self._index
    self.images = [i.info for i in tmp]
    self.save_meta(meta_path)
    self.images = tmp
    self._index = index
    tar = tarfile.open(self._full_data_path(meta_path), 'w')
    for img in self.images:
      tmp_path = tempfile.mkstemp()[1]
      img.save(tmp_path, img.format)
      arc_path = '%s/%s.%s' % (self.id, img.info['id'],
                               img.info['format'].lower())
      tar.add(tmp_path, arcname=arc_path)
      os.unlink(tmp_path)
    tar.close()

  def load(self, meta_path=None):
    """load estim from filesystem"""
    log.info("loading estim from %s", meta_path)
    self.load_meta(meta_path)
    self._index = {}
    imgs_meta = self.images[:]
    log.debug("   '%s': %d images", self.name, len(imgs_meta))
    tar = tarfile.open(self._full_data_path(meta_path))
    self.images = []
    for img_meta in imgs_meta:
      log.debug("   loading image: %s", img_meta['id'])
      arc_path = '%s/%s.%s' % (self.id, img_meta['id'],
                               img_meta['format'].lower())
      img = Image.open(tar.extractfile(arc_path))
      img.load()
      img.info = img_meta
      self.images.append(img)
      self._index[img_meta['id']] = img

  def add_image(self, path, source=None, date=None, resize=1024):
    """add an image with metadata to the collection"""
    log.debug("   adding image: %s", path)
    if source is None: source = os.path.abspath(path)
    if date   is None: date   = utils.datestr(os.stat(path).st_ctime)
    fname   = os.path.basename(path)
    imgnum  = '%03d' % (self.images
                        and max(int(i.info['id'][-3:]) for i in self.images) + 1
                        or 1)
    img     = Image.open(path)
    if resize:
      w, h = img.size
      new_w, new_h = img.size
      if w > resize:
        new_w = 1024
        new_h = utils.iround((h * 1024) / w)
      if h > resize and new_h > resize:
        new_w = utils.iround((w * 1024) / h)
        new_h = 1024
      if new_w != w or new_h != h:
        log.debug("   resizing: %dx%d -> %dx%d", w, h, new_w, new_h)
        img.resize((new_w, new_h), Image.ANTIALIAS)
    rdx_img = utils.reduce_image(img, 5, 5, 5)
    redux   = ''.join('%d' % x for x
                      in (v / 255 * 5 for v in  rdx_img.getdata()))
    img.info = {'name':   fname,
                'id':     self.id + '.' + imgnum,
                'source': str(source),
                'date':   str(date),
                'mode':   img.mode,
                'format': img.format,
                'width':  img.size[0],
                'height': img.size[1],
                'redux':  redux}
    self.images.append(img)
    self._index[img.info['id']] = img

  def get_image(self, id):
    """image by id"""
    return self._index[id]

  def _full_data_path(self, meta_path):
    """build a full path for the data file"""
    return os.path.join(os.path.dirname(meta_path), self.data_path)

#------------------------------------------------------------------------------
if __name__ == '__main__':
  from oe3.utils import run_doctest
  run_doctest('coag_estim')
