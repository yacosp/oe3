# oe3 lib/oe3/viewer.py
#
# oveja electrica - oe3.0.1:bleu
# copyright (c) 2003-2017 santiago pereson <yaco@yaco.net>
# http://yaco.net/oe/
#
# this file is part of 'oe3', which is released under the gnu general public
# license, version 3. see LICENSE for full license details.
#

"""oe3 simple viewer class"""


from __future__ import division

import logging

from PIL import Image

from oe3 import oe3_path, utils


conf_path = oe3_path + '/etc/viewer.conf'
log = logging.getLogger('oe3')


#------------------------------------------------------------------------------
class Viewer(object):
  """stub"""

  def __init__(self):
    """read conf"""
    self.conf = utils.load_dict(conf_path)

  def gallery(self, images, show_images=False):
    """generate a gallery for an image collection"""
    gutter = self.conf['gallery_gutter']
    num_images = len(images)
    i = 0
    while self.conf['gallery_thumb_side'][i][0] < num_images: i += 1
    gal_len, gal_cols, gal_rows, thw = \
             self.conf['gallery_thumb_side'][i]
    gal = Image.new('RGB', self.conf['gallery_size'])
    for i, img in enumerate(images):
      col, row = i %  gal_cols, i // gal_cols
      x = gutter // 2 + col * (thw + gutter)
      y = gutter // 2 + row * (thw + gutter)
      iw, ih = img.size
      ratio = thw / max([iw, ih])
      thumb = img.resize(
          (int(iw * ratio), int(ih * ratio)), Image.ANTIALIAS)
      if show_images:
        tw, th = thumb.size
        gal.paste(thumb, (x + (thw - tw) // 2, y + (thw - th) // 2))
      else:
        gal.paste(
            utils.reduce_image(thumb, 5, 5, 5).resize((thw, thw)),
            (x, y))
    return gal

#------------------------------------------------------------------------------
if __name__ == '__main__':
  run_doctest('viewer')
