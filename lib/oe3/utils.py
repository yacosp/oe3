# oe3 lib/oe3/utils.py
#
# oveja electrica - oe3.0.1:bleu
# copyright (c) 2003-2017 santiago pereson <yaco@yaco.net>
# http://yaco.net/oe/
#
# this file is part of 'oe3', which is released under the gnu general public
# license, version 3. see LICENSE for full license details.
#

"""oe3 utility methods"""


from __future__ import division

import bz2
import colorsys
import logging
import math
import os
import ossaudiodev
import pprint
import random
import time
import wave

from commands import getstatusoutput
from glob     import glob
from shutil   import copyfileobj

from PIL import Image
from PIL import ImageOps

from oe3 import oe3_path


__version__ = '0.1.3'
log = logging.getLogger('oe3')


# data structure stuff --------------------------------------------------------
def max_key(d):
  """return the key corresponding to the max value in a dict"""
  maxval = max(d.values())
  keys   = (i[0] for i in d.iteritems() if i[1] == maxval)
  return keys.next()

def hml_find_limits(vals):
  """find zone limits for a list if integer values.
  returns a tuple: (<low>,<mid_low>,<mid_high>,<high>)
  """
  low     = int(min(vals))
  high    = int(max(vals))
  third   = (high - low) / 3
  midlow  = int(math.floor(low + third))
  midhigh = int(math.ceil(high - third))
  return (low, midlow, midhigh, high)

def hml_value_limits(limits, val, fullrange=(0,100)):
  """find which limits contain the specified value"""
  if val < limits[0]:
    return (fullrange[0], limits[0])
  else:
    for i in range(len(limits) - 1):
      if val < limits[i + 1]:
        return (limits[i], limits[i + 1])
  return (limits[-1], fullrange[-1])

def hml_split_list(vals):
  """
  split a value list in three.

  ie: [1,2,3,4,5,6,7] => [[1,2], [3,4,5], [6,7]]
  """
  cnt   = len(vals)
  third = cnt / 3
  midlo = int(math.floor(third))
  midhi = int(math.ceil(third * 2))
  return [vals[:midlo], vals[midlo:midhi], vals[midhi:]]

def hml_find_zone(vals, val):
  """find zone (hi: +1, mid: 0, lo: -1) in which a value is found in vals"""
  zlists = hml_split_list(vals)
  idx    = -1
  for zlist in zlists:
    if val in zlist:
      return idx
    idx += 1
  return None

# file stuff ------------------------------------------------------------------
# XXX to-do: use json: www.json.org
def load_dict(path):
  """load a dict from a text file"""
  return eval(open(path, 'r').read())

def save_dict(path, dct, backup=False):
  """save a dict to a text file"""
  if not pprint.isreadable(dct):
    raise ValueError("dict contains non-printable data")
  if backup and os.path.isfile(path): os.rename(path, path + '~')
  open(path, 'w').write(pprint.pformat(dct))

def load_bzdict(path):
  """load a dict from a bz2 compressed text file"""
  return eval(bz2.BZ2File(path, 'r').read())

def save_bzdict(path, dct, backup=False):
  """save a dict to a bz2 compressed text file"""
  if not pprint.isreadable(dct):
    raise ValueError("dict contains non-printable data")
  if backup and os.path.isfile(path): os.rename(path, path + '~')
  bz2.BZ2File(path, 'w').write(pprint.pformat(dct))

def compress_logs():
  """compress old logs"""
  logs = glob(
    oe3_path + '/var/log/oe3.log.[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
  )
  if len(logs):
    log.info(u"compressing logs...")
    for logfile in logs:
      with open(logfile, 'rb') as infile:
        with bz2.BZ2File(logfile + '.bz2', 'wb') as outfile:
          copyfileobj(infile, outfile)
      os.remove(logfile)

def int2le16(val):
  """python 16-bit int -> 16bit little-endian data (str)"""
  return chr((val  % 256) & 255) + chr((val // 256) & 255)

# image stuff -----------------------------------------------------------------
def reduce_image(image, width, height, levels=None, normalize=True):
  """convert to grayscale, rescale, posterize and normalize a PIL image"""
  temp = image.convert('L').resize((width,height), Image.ANTIALIAS)
  if normalize:
    temp = ImageOps.autocontrast(temp)
  if levels is not None:
    temp = temp.point(_reduce_depth_table(levels))
  return temp

def _reduce_depth_table(levels):
  """posterize filter table for a grayscale image .point() function"""
  # XXX this is reducing to levels + 1 ?
  lstep = 256 / levels
  vstep = 255 / (levels - 1)
  ft    = []
  for i in range(levels):
    ft += [int(vstep * i)] * int(lstep)
    if lstep * (i + 1) > len(ft): ft += [int(vstep * i)]
  return ft

def rgb2gray16(r, g, b):
  """rgb to 16-bit grayscale conversion"""
  return int(r * 76.544 + g * 150.272 + b * 29.184)

def rgb2hsl(r, g, b):
  """colorsys.rgb_to_hls stub"""
  h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
  return h, s, l

def image_hsl_split(image):
  """generate HSL grayscale images from a color image"""
  hband, sband, lband = [], [], []
  for r, g, b in list(image.getdata()):
    h, s, l = rgb2hsl(r, g, b)
    hband.append(iround(h * 255))
    sband.append(iround(s * 255))
    lband.append(iround(l * 255))
  hi, si, li = (Image.new('L', image.size),
                Image.new('L', image.size),
                Image.new('L', image.size))
  hi.putdata(hband)
  si.putdata(sband)
  li.putdata(lband)
  return hi, si, li

# math stuff ------------------------------------------------------------------
def distance_n1(x, y):
  """calculate 1-norm distance between two equal-length integer sequences"""
  dist = 0
  for i in range(len(x)):
    dist += abs(abs(int(x[i])) - abs(int(y[i])))
  return dist

def iround(val):
  """rounded integer"""
  return int(round(val))

def mean(lst):
  """arithmetic mean"""
  return sum(lst) / len(lst)

def rand_below(val):
  """True if a random value is below <val>, which should be in [0, 100]"""
  return random.randint(0, 100) <= val

def stdev(lst):
  """standard deviation"""
  llen  = len(lst)
  lmean = mean(lst)
  var   = sum((lst[i] - lmean) * (lst[i] - lmean)
              for i in range(llen)) / (llen - 1)
  return math.sqrt(var)

# sound stuff -----------------------------------------------------------------
def hz2px(hz, px_range, hz_range=(20, 24000)):
  """hertz to pixels conversion"""
  min_px, max_px = px_range
  min_hz, max_hz = hz_range
  a  = (max_px - min_px) / math.log10(max_hz / min_hz)
  px = a * math.log10(hz / min_hz) + min_px
  return iround(px)

def px2hz(px, px_range, hz_range=(20, 24000)):
  """pixels to hertz conversion"""
  min_px, max_px = px_range
  min_hz, max_hz = hz_range
  a  = (max_px - min_px) / math.log10(max_hz / min_hz)
  hz = min_hz * pow(10, (px - min_px) / a)
  return iround(hz)

def play_sound(sfile, dur=None):
  """play the first <dur> seconds of the specified wav file"""
  sf  = wave.open(sfile, 'rb')
  ad  = ossaudiodev.open('w')
  fmt = [ossaudiodev.AFMT_S8, ossaudiodev.AFMT_S16_LE][sf.getsampwidth() - 1]
  sr  = sf.getframerate()
  ad.setparameters(fmt, sf.getnchannels(), sr)
  if dur is None:
    nframes = sf.getnframes()
  else:
    nframes = int(dur * sr)
  ad.write(sf.readframes(nframes))
  sf.close()
  ad.close()

# testing stuff ---------------------------------------------------------------
def run_doctest(test):
  """run one module's doctest"""
  import doctest, os
  os.chdir(os.path.join(oe3_path + '/lib'))
  f, c = doctest.testfile(
    os.path.join(oe3_path, 'lib/oe3/test/%s.doctest' %  test),
    module_relative=False)
  if f == 0: print 'no errors in', c, 'tests.'

# time stuff ------------------------------------------------------------------
def datestr(secs=None):
  """return a formatted time string, ie: '2005-10-25 05:32:01'"""
  if secs is None: secs = time.time()
  return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(secs))

def tstamp(secs=None, sep='.'):
  """return a timestamp string, ie: '20051025.053201'"""
  if secs is None: secs = time.time()
  return time.strftime('%Y%m%d' + sep + '%H%M%S', time.gmtime(secs))

# other stuff -----------------------------------------------------------------
def shexec(cmd, *args, **kwargs):
  """
  execute the command, returning stdin & stderr output

  if **kwargs are provided, *args will not be parsed.
  raises an exception if the command's status is not 0.
  """
  if kwargs:
    cmd = cmd % kwargs
  elif args:
    cmd = cmd % args
  s, o = getstatusoutput(cmd)
  if s != 0:
    raise EnvironmentError(s, o)
  return o

# main ------------------------------------------------------------------------
if __name__ == '__main__':
  run_doctest('utils')
