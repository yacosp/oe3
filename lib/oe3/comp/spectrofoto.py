# oe3 lib/oe3/comp/spectrofoto.py
#
# oveja electrica - oe3.0.1:bleu
# copyright (c) 2003-2017 santiago pereson <yaco@yaco.net>
# http://yaco.net/oe/
#
# this file is part of 'oe3', which is released under the gnu general public
# license, version 3. see LICENSE for full license details.
#

"""spectrofoto subcomposer class and helper functions"""


from __future__ import division

import logging
import os
import pdb
import random
import re
import shutil
import sys
import tarfile
import time
import wave
from math import ceil, pi, sin
from operator import itemgetter

from PIL import Image
from PIL import ImageFilter
from PIL import ImageStat
import numpy

from oe3 import oe3_path, utils
from oe3.crit import MAX_REWARD
from oe3.opus import song
from oe3.utils import iround, mean, shexec, stdev


__version__ = '0.1.2'

conf_path = os.path.join(oe3_path, 'etc/spectrofoto.conf')
log = logging.getLogger('comp')


#------------------------------------------------------------------------------
class Spectrofoto(object):
  """an 'image group to quasi spectral reconstruction' subcomposer"""

  # main stuff ----------------------------------------------------------------
  def create(self, estim, seed=None, choices=None):
    """create a new composition"""
    log.info("creating song from estim '%s' (%s)", estim.name, estim.id)
    try:
      self._init_song(estim, seed)
      self._analyze_estim()
      self._choose_options(choices)
      self._create_song()
      self._wrap_song()
    except Exception, e:
      log.critical(e)
      raise
    return self.song

  def learn(self, song, reward):
    """learn from a reward"""
    log.info("learning reward %s for song '%s'", reward, song.id)
    self.state['histo'].append((song.id, song.estim.id, song.anal.copy(),
                                song.choices.copy(), reward))
    self.save_state()

  # init song -----------------------------------------------------------------
  def _init_song(self, estim, seed=None):
    """initialize new song"""
    if seed is None: seed = time.time()
    log.info("initializing new song with estim: %s and seed: %s",
             estim.id, seed)
    self.song = song.Song(estim, (self.__class__.__name__, __version__))
    log.debug("   song id is %s", self.song.id)
    self.song.seed = seed
    random.seed(self.song.seed)
    self.song.images = [i.resize((192, 192), Image.ANTIALIAS)
                        for i in self.song.estim.images]
    self.song.img_medians = {}
    self._save_runfile(5)
    self.song.tmpd = os.path.join(oe3_path, 'tmp/comp', self.song.id + '.tmp')
    os.mkdir(self.song.tmpd)
    os.chdir(self.song.tmpd)


  # analyze estim -------------------------------------------------------------
  def _analyze_estim(self):
    """
    get (min, max, range, mean, stdev) of
    (r, g, b, l, l_top, l_mid, l_bottom, distance)
    """
    log.info("analyzing estim")
    aspects = ('red', 'grn', 'blu', 'gry', 'top', 'mid', 'btm')
    self.song.anal = {'num_images': len(self.song.images)}
    log.debug("   %s", self.song.anal)
    temp = {}
    for aspect in aspects:
      temp = []
      for img in self.song.images:
        if img.info['mode'] != 'RGB':
          # XXX no funca
          img = img.convert('RGB')
        temp.append(self._analyze_image_aspect(img, aspect))
      self.song.anal.update(self._reduce_anal_stats(temp, aspect))
    self.song.anal.update(self._analyze_estim_distances())
    utils.save_bzdict('anal.pyd.bz2', self.song.anal)

  def _analyze_image_aspect(self, img, aspect):
    """analyze one image aspect"""
    ih, iw = img.size
    ih3 = ih / 3
    if   aspect == 'red':
      chan = img.split()[0]
    elif aspect == 'grn':
      chan = img.split()[1]
    elif aspect == 'blu':
      chan = img.split()[2]
    elif aspect == 'gry':
      chan = img.convert('L')
    elif aspect == 'top':
      chan = img.convert('L').crop((0, 0, iw, int(ih3)))
    elif aspect == 'mid':
      chan = img.convert('L').crop((0, int(ih3), iw, int(ceil(2 * ih3))))
    elif aspect == 'btm':
      chan = img.convert('L').crop((0, int(ceil(2 * ih3)), iw, ih))
    stat = ImageStat.Stat(chan)
    imin, imax = stat.extrema[0]
    if aspect == 'gry':
      self.song.img_medians[img.info['id']] = stat.median[0]
    return {'min':   iround((imin / 255) * 100),
            'max':   iround((imax / 255) * 100),
            'delta': iround(((imax - imin) / 255) * 100),
            'mean':  iround((stat.mean[0] / 255) * 100),
            'stdev': iround((stat.stddev[0] / 255) * 100)}

  def _reduce_anal_stats(self, allstats, aspect):
    """reduce one aspect stats for all images"""
    stats = {}
    for stat in ('min', 'max', 'delta', 'mean', 'stdev'):
      stats['%s_%s' % (aspect, stat)] = \
          iround(mean([x[stat] for x in allstats]))
    log.debug("   %s", sorted(stats.iteritems()))
    return stats

  def _analyze_estim_distances(self):
    """
    analyze 1-norm distance between all images,
    find anomaly, generate distance stats
    """
    num_images, distances, sums = len(self.song.images), [], []
    for i in range(num_images):
      distances.append([])
      for j in range(num_images):
        dist = utils.distance_n1(self.song.images[i].info['redux'],
                                 self.song.images[j].info['redux'])
        distances[i].append(dist)
      sums.append(sum(distances[i]))
    anom_idx   = sums.index(max(sums))
    anom_dists = distances[anom_idx]
    self.song.img_anomaly = self.song.images[anom_idx].info['id']
    self.song.img_distances = \
        dict((self.song.images[i].info['id'], anom_dists[i])
             for i in range(num_images))
    del anom_dists[anom_idx]   # anomaly not included in stats
    amin = min(anom_dists)
    amax = max(anom_dists)
    stats = {'dst_min':   amin,
             'dst_max':   amax,
             'dst_delta': amax - amin,
             'dst_mean':  int(mean(anom_dists)),
             'dst_stdev': int(stdev(anom_dists))}
    log.debug("   %s", sorted(stats.iteritems()))
    log.debug("   anomaly is %s", self.song.img_anomaly)
    return stats


  # choose options ------------------------------------------------------------
  def _choose_options(self, choices=None):
    """choose parameters for each option"""
    self._anal_choices()
    self._intra_choices()
    self._exm_choices()
    self._fixed_choices(choices)
    utils.save_bzdict('choices.pyd.bz2', self.song.choices)

  # anal choices --------------------------------------------------------------
  def _anal_choices(self):
    """choose parameters based on estim analysis and experience"""
    log.info("choosing parameters based on estim analysis")
    choices = {}
    for option, values in self.conf['choice_tpl'].iteritems():
      votes = dict().fromkeys(values, 0)
      certs = dict([(val, []) for val in values])
      for aspect in self.song.anal.iterkeys():
        choice, cert = \
            self._anal_choose(option, values, aspect, self.song.anal[aspect])
        votes[choice] += 1
        certs[choice].append(cert)
      winner = utils.max_key(votes)
      choices[option] = (winner, int(utils.mean(certs[winner])))
      log.debug("   option: '%s', choice: %s;\n votes: %s\n certs: %s",
                option, choices[option], votes, certs)
    self.song.choices = choices
    log.info("anal: %s", pformat_choices(self.song.choices))
    self._save_runfile(10)

  def _anal_choose(self, option, values, aspect, anal_value):
    """
    choose between possible option values historically
    associated with one analysis aspect
    """
    ahist = self._aspect_history(aspect, option)
    ahist_len = len(ahist)
    if ahist_len < len(values):
      rchoice = random.choice(values)
      return (rchoice, 0)  # shortcircuit: explore
    lims = utils.hml_find_limits([i[0] for i in ahist])
    zmin, zmax = utils.hml_value_limits(lims, anal_value)
    relevant = dict().fromkeys(values, 0)
    max_rwd  = dict().fromkeys(values, 0)
    for i, (old_val, old_choice, old_reward) in enumerate(ahist):
      if old_val >= zmin and old_val < zmax:
        relevant[old_choice] += ((i + 1) / ahist_len) * old_reward
        max_rwd[old_choice]  += ((i + 1) / ahist_len) * MAX_REWARD
    choice = utils.max_key(relevant)
    if max_rwd[choice] > 0:
      tried  = [val for val in values if relevant[val] > 0]
      tryall = len(tried) / len(values)
      cert   = int((relevant[choice] / max_rwd[choice]) * tryall * 100)
    else:
      cert = 0
    if not utils.rand_below(cert):  # not enough cert. explore!
      rchoice = random.choice(values)
      if rchoice != choice:
        choice, cert = rchoice, 0
    return (choice, cert)

  # intra choices -------------------------------------------------------------
  def _intra_choices(self):
    """alter choices based on learnt internal option relationships"""
    log.info("altering parameters based on learnt internal relationships")
    foption, fzlist = self._intra_find_focus()
    if foption is None: return
    chist = self._choice_history(foption, fzlist)
    self._intra_modify_choices(foption, chist)
    log.info("intra: %s", pformat_choices(self.song.choices))
    self._save_runfile(13)

  def _intra_find_focus(self):
    """find choice with highest cert"""
    choice_certs = dict((option, choice[1])
                        for option, choice in self.song.choices.iteritems())
    foption = utils.max_key(choice_certs)
    fchoice, fcert = self.song.choices[foption]
    log.debug("   focus ch.: %s: %s (%s)", foption, fchoice, fcert)
    if not utils.rand_below(fcert):
      log.debug("   low focus cert : %s. leaving choices alone.", fcert)
      return None, None  # shortcircuit
    fzone = utils.hml_find_zone(self.conf['choice_tpl'][foption], fchoice)
    if fzone == 0:
      log.debug("   focus is in middle zone. leaving choices alone.")
      return None, None  # shortcircuit
    fzlist = utils.hml_split_list(
        self.conf['choice_tpl'][foption])[fzone + 1]
    log.debug("   focus zone: %+d; zone choices: %s", fzone, fzlist)
    return foption, fzlist

  def _intra_modify_choices(self, foption, chist):
    """modify choices"""
    for option, values in self.conf['choice_tpl'].iteritems():
      if option != foption:
        zone = utils.hml_find_zone(values, self.song.choices[option][0])
        if zone != 0:
          rankg = self._intra_choice_ranking(chist, option)
          winzone = rankg[0][0]
          log.debug("   choice %15s; cur: %+d; rankg: %s", option, zone, rankg)
          if zone + winzone == 0:  # if opposed (+1 + -1 == 0, rest != 0)
            # invert choice
            idx = values.index(self.song.choices[option][0])
            self.song.choices[option] = (values[-(idx + 1)], rankg[0][2])
            log.debug("   inverted choice: %s", self.song.choices[option])

  def _intra_choice_ranking(self, chist, option):
    """rank previous choices for <option>"""
    # ranking: [(<zone>, <aged_reward_sum>, <cert>), ...]
    rankg_tmp = {1: [0, 0], 0: [0, 0], -1: [0, 0]}
    chist_len = len(chist)
    for i in range(chist_len):
      choices, reward = chist[i]
      zone = utils.hml_find_zone(self.conf['choice_tpl'][option],
                                 choices[option][0])
      rankg_tmp[zone][0] += ((i + 1) / chist_len) * reward
      rankg_tmp[zone][1] += ((i + 1) / chist_len) * MAX_REWARD
    rankg = []
    for zone, rewds in rankg_tmp.iteritems():
      if rewds[1] > 0:
        # XXX add tryall ratio?
        cert = int((rewds[0] / rewds[1]) * 100)
      else:
        cert = 0
      rankg.append((zone, rewds[0], cert))
    rankg.sort(key=itemgetter(1), reverse=True)
    return rankg

  # exm_choices ---------------------------------------------------------------
  def _exm_choices(self):
    """alter choices randomly"""
    log.info("altering low cert parameters at random")
    min_cert = min(choice[1] for choice in self.song.choices.itervalues())
    thresh = max(self.conf['exm_thresh'], min_cert)
    log.debug("   exm threshold: %s", thresh)
    for option, choice in self.song.choices.iteritems():
      if choice[1] <= thresh and choice[1] != 0:  # 0-cert == random
        rchoice = random.choice(self.conf['choice_tpl'][option])
        if rchoice != choice[0]:
          self.song.choices[option] = (rchoice, 0)
          log.debug("   %s: %s (%s) -> %s: %s",
                    option, choice[0], choice[1], option, rchoice)
    log.info("exm: %s", pformat_choices(self.song.choices))
    self._save_runfile(17)

  # fixed_choices -------------------------------------------------------------
  def _fixed_choices(self, choices):
    """set fixed choices"""
    self._fix_song_dur()
    if choices is not None:
      for k, v in choices.iteritems():
        if v.isdigit(): v = int(v)
        self.song.choices[k] = (v, 0)
      log.info("fixed: %s", pformat_choices(self.song.choices))

  def _fix_song_dur(self):
    """modify sect_base_time so song duration is within bounds"""
    # XXX this should be cleaned up! + check that -m _without_ -M works!
    len_estim = len(self.song.estim.images)
    sbt_vals  = self.conf['choice_tpl']['sect_base_time']
    log.debug("   len_estim: %d; sbt: %d; result dur: %d",
              len_estim, self.song.choices['sect_base_time'][0],
              self.song.choices['sect_base_time'][0] * len_estim)
    min_sd = self.conf['min_song_dur']
    max_sd = self.conf['max_song_dur']
    if min_sd is None and max_sd is None: return   # short
    log.debug("   min_sd: %d; max_sd: %d", min_sd, max_sd)
    try:
      while self.song.choices['sect_base_time'][0] * len_estim < min_sd:
        idx = sbt_vals.index(self.song.choices['sect_base_time'][0])
        log.warning("   forced min song duration, fixing sbt. %s -> %s",
                    sbt_vals[idx], sbt_vals[idx + 1])
        self.song.choices['sect_base_time'] = (sbt_vals[idx + 1], 0)
    except IndexError:
      log.warning("   cannot reduce sbt any more!")
    try:
      while self.song.choices['sect_base_time'][0] * len_estim > max_sd:
        idx = sbt_vals.index(self.song.choices['sect_base_time'][0])
        log.warning("   forced max song duration, fixing sbt. %s -> %s",
                    sbt_vals[idx], sbt_vals[idx - 1])
        self.song.choices['sect_base_time'] = (sbt_vals[idx - 1], 0)
    except IndexError:
      log.warning("   cannot increment sbt any more!")
    # XXX other fixed choices?
    self._save_runfile(20)


  # create song ---------------------------------------------------------------
  def _create_song(self):
    """create a song based on composer choices"""
    log.info("rendering song")
    choices = dict((k, v[0]) for k, v in self.song.choices.iteritems())
    self._resort_images(choices)
    trackl = self._prep_tracks(choices)
    self._save_runfile(42)
    self._gen_sources(trackl)
    mixl = self._filter_sections(trackl)
    self._mix_tracks(mixl)

  # resort images -------------------------------------------------------------
  def _resort_images(self, choices):
    """reorder a copy of the estim image sequence"""
    log.info("modifying image collection: %s, %s, %s",
             choices['sort'], choices['randomize'], choices['transform'])
    getattr(self, '_resort_%s' % choices['sort'])()
    log.debug("   sorted:      %s",
              [img.info['id'][-3:] for img in self.song.images])
    self._save_runfile(22)
    getattr(self, '_resort_%s' % choices['randomize'])()
    log.debug("   randomized:  %s",
              [img.info['id'][-3:] for img in self.song.images])
    self._save_runfile(24)
    getattr(self, '_resort_%s' % choices['transform'])()
    log.debug("   transformed: %s",
              [img.info['id'][-3:] for img in self.song.images])
    self.song.img_idx = [img.info['id'] for img in self.song.images]
    self._save_runfile(26)

  def _resort_none(self): pass

  def _resort_median_l(self):
    """reorder images based on median lightness"""
    self.song.images.sort(self._median_sorter)
    log.debug("   medians: %s",
              [(i.info['id'][-3:], self.song.img_medians[i.info['id']])
               for i in self.song.images])

  def _median_sorter(self, x, y):
    """sort by median image value"""
    return cmp(self.song.img_medians[x.info['id']],
               self.song.img_medians[y.info['id']])

  def _resort_dist_n1(self):
    """reorder images based on distance to anomaly"""
    self.song.images.sort(self._dist_n1_sorter)
    log.debug("   distances: %s",
              [(i.info['id'][-3:], self.song.img_distances[i.info['id']])
               for i in self.song.images])

  def _dist_n1_sorter(self, x, y):
    """sort by norm-1 distance"""
    return cmp(self.song.img_distances[x.info['id']],
               self.song.img_distances[y.info['id']])

  def _resort_gaussian(self):
    """select images based on a gaussian distribution"""
    resorted = []
    num_img = len(self.song.images)
    mu, sigma = num_img / 2, num_img / 4
    for i in range(num_img):
      img = self.song.images[int(min(num_img - 1, random.gauss(mu, sigma)))]
      resorted.append(img)
    self.song.images = resorted

  def _resort_maper(self):
    """random walk on an image pizza"""
    # XXX to be done
    pass

  def _resort_shuffle(self):
    """shuffle the collection"""
    random.shuffle(self.song.images)

  def _resort_uniform(self):
    """select images based on a uniform distribution"""
    resorted = []
    for i in range(len(self.song.images)):
      resorted.append(random.choice(self.song.images))
    self.song.images = resorted

  def _resort_reverse(self):
    """reverse image order"""
    self.song.images = self.song.images[::-1]

  def _resort_mirror(self):
    """mirror image order"""
    if len(self.song.images) % 2 == 0:
      mirror = self.song.images[::-2]
    else:
      mirror = self.song.images[-2::-2]
    self.song.images = self.song.images[::2] + mirror

  # prepare tracks ------------------------------------------------------------
  def _prep_tracks(self, choices):
    """prepare mixlist for each track"""
    log.info("preparing tracks")
    log.info("choices: %s", pformat_choices(self.song.choices))
    hsls = self._all_hsl_images(self.song.images)
    trackl = {}
    max_dur = 0
    for timbre, stype in zip('bps', choices['spectrum_source']):
      if timbre in choices['timbres']:
        log.debug("   preparing track '%s' (source: '%s')", timbre, stype)
        src_chan = self._timbre_channel(timbre)
        shared_src = self._shared_source(
            stype=stype, images=hsls[src_chan], channel=src_chan)
        each_p = stype == 'e'
        dur, sections = self._prep_sections(choices=choices,
                                            images=hsls[src_chan],
                                            each_p=each_p, channel=src_chan)
        max_dur = max(max_dur, dur)
        trackl[timbre] = {'img': shared_src,
                          'snd': None,
                          'dur': dur,
                          'sections': sections}
    for timbre in choices['timbres']:
      trackl[timbre]['start'] = self._track_start(align=choices['sect_align'],
                                                  dur=trackl[timbre]['dur'],
                                                  max_dur=max_dur,
                                                  hues=hsls['h'])
      self.song.tracklist = trackl
    log.debug("   saving tracklist ...")
    utils.save_bzdict('tracklist.pyd.bz2', trackl)
    self.song.hsls = hsls
    return trackl

  def _prep_sections(self, choices, images, each_p, channel):
    """prepare one track's sections"""
    sections = []
    offset = 0
    for img in images:
      dur = (choices['sect_base_time'] *
             self._calc_sdr(img, choices['sect_dur_ratio']))
      redux = self._section_image_redux(img, choices['env_follow'], dur)
      log.debug("   redux width: %d", redux.size[0])
      bands = self._prep_section_bands(redux, dur)
      if each_p:
        src = (img.info['id'], channel)
      else:
        src = None
      sections.append({'start': offset,
                       'dur': dur,
                       'img': src,
                       'snd': None,
                       'bands': bands})
      offset += dur
      log.debug("   %2d: %7.3fs @ %7.3f, %d bands.",
                len(sections), dur, offset, len(bands))
    return (offset, sections)

  def _prep_section_bands(self, img, dur):
    """prepare one section's band mix lists"""
    bands = []
    imgw = img.size[0]
    ibands = self._image_split_bands(img, self.conf['bpf_range'])
    # reversed so low freqs get filtered first
    for center, width, bimg in ibands[::-1]:
      amps = self._prep_band_amps(bimg, dur)
      if amps is not None:
        bands.append({'center': center, 'width': width, 'amps': amps})
    return bands

  def _prep_band_amps(self, img, dur):
    """prepare one band's stereo amp list"""
    imgw = img.size[0]
    pan_data = list(img.resize((imgw, 2), Image.ANTIALIAS).getdata())
    if len(pan_data) == 0 or max(pan_data) == 0:  # shortcircuit
      log.debug("   skipping black band")
      return None
    lvals, rvals = pan_data[:imgw], pan_data[imgw:]
    if imgw == 1:
      step = dur
      lvals.append(lvals[0])
      rvals.append(rvals[0])
    else:
      step = dur / (imgw - 1)
    amps = []
    namps = len(lvals)
    diffl  = [lvals[i] - rvals[i] for i in range(namps)]
    vdelta = abs(min(diffl)) > abs(max(diffl)) and min(diffl) or max(diffl)
    maxamp, lprev, rprev = 0, 0, 0
    for i in range(namps):
      if vdelta == 0:
        lamp = lvals[i] / 255
        ramp = rvals[i] / 255
      else:
        vavg = (lvals[i] + rvals[i]) / 2
        pan  = (((lvals[i] - rvals[i]) / vdelta) + 1) / 2
        lamp = (vavg * (1 - pan)) / 255
        ramp = (vavg * pan) / 255
      # only add env step if it's the first, last, or has a different value
      if (lamp != lprev and ramp != rprev) \
             or i == 0 or i == namps - 1 \
             or (lvals[i] != lvals[i + 1] or rvals[i] != rvals[i + 1]):
        amps.append((i * step, lamp, ramp))
        lprev, rprev = lamp, ramp
        maxamp = max(maxamp, lamp, ramp)
    if maxamp < self.conf['amp_min']:
      log.debug("   skipping almost black band")
      amps = None
    return amps

  def _all_hsl_images(self, images):
    """generate hsl images for each rgb one"""
    # XXX to-do: cache splits
    hsl  = {'h': None, 's': None, 'l': None}
    hsls = {'h': [], 's': [], 'l': []}
    for img in images:
      log.debug("   splitting '%s' hsl channels", img.info['id'])
      hsl['h'], hsl['s'], hsl['l'] = utils.image_hsl_split(img)
      if self.conf['hsl_gray_kluge']:
        if max(hsl['h'].getdata()) < 11:
          hsl['h'] = hsl['l']
          log.debug("   hsl grayscale kluge: channel 'H' copied from 'L'!")
        if max(hsl['s'].getdata()) < 11:
          hsl['s'] = hsl['l']
          log.debug("   hsl grayscale kluge: channel 'S' copied from 'L'!")
      for channel in 'hsl':
        hsl[channel].info = dict(img.info)
        hsls[channel].append(hsl[channel])
    return hsls

  def _timbre_channel(self, timbre):
    """select hsl channel for a timbre"""
    tcmap = self.conf['timbre_channels']
    tchans = tcmap[self.song.choices['timbres'][0]]
    return tchans[timbre]

  def _shared_source(self, stype, images, channel):
    """shared source image for one track"""
    if stype == 'e':
      return None
    else:
      clist = [(img.info['id'], self._image_complexity(img)) for img in images]
      clist.sort(key=itemgetter(1))
      if   stype == 'c': img_id = clist[-1][0]
      elif stype == 's': img_id = clist[0][0]
    return (img_id, channel)

  def _image_complexity(self, img, redux=False):
    """find out the level of complexity of an image"""
    if redux: img = img.convert('L').resize((96, 96), Image.ANTIALIAS)
    histo = img.filter(ImageFilter.CONTOUR).convert('1').histogram()
    return (histo[0] / (histo[0] + histo[-1])) * 2

  def _calc_sdr(self, img, sdr):
    """calc sect_dur_ratio"""
    if sdr == 'fixed':
      ratio = 0.5
    elif sdr in ('median', 'inv_median'):
      ratio = ImageStat.Stat(img).median[0] / 255
      ratio = sdr == 'inv_median' and 1 - ratio or ratio
    elif sdr in ('prop', 'inv_prop'):
      peaks = self._image_peaks(img.resize((img.size[0],1), Image.ANTIALIAS))
      ratio = len(peaks) / 48
      ratio = sdr == 'inv_prop' and 1 - ratio or ratio
    return 0.5 + ratio

  def _image_peaks(self, img):
    """find peaks in an image"""
    data  = img.getdata()
    lenm1 = len(data) - 1
    peaks, nadirs = [], []
    if data[0] >= data[1]: peaks.append(0)
    else: nadirs.append(0)
    for i in range(1, lenm1):
      if data[i] > data[i - 1] and data[i] >= data[i + 1]:
        peaks.append(i)
      if data[i] < data[i - 1] and data[i] <= data[i + 1]:
        nadirs.append(i)
    if data[-2] < data[-1]: peaks += [lenm1]
    else: nadirs += [lenm1]
    return peaks

  def _section_image_redux(self, img, efl, dur):
    """image redux for one section, according to env_follow and duration"""
    if efl == 1:
      imgw = 1
    else:
      imgw = iround((efl - 1) * img.size[0] * 0.1)
    imgw = min(imgw, iround(dur * self.conf['max_amps_per_sec']))
    return img.resize((imgw, img.size[1]), Image.ANTIALIAS)

  def _image_split_bands(self, img, range_hz=(20, 24000)):
    """split an image in peak-centered bands"""
    imgh = img.size[1]
    if img.size[0] == 1:
      redux = img.copy()
    else:
      redux = img.resize((1, imgh), Image.ANTIALIAS)
    peaks = self._image_peaks(redux)
    len_peaks = len(peaks)
    if len_peaks == 1:
      return [self._one_band_item(peaks[0], (0, imgh), range_hz, img)]
    elif len_peaks > self.conf['max_bands']:
      peaks = self._reduce_peaks(peaks, redux)
      len_peaks = len(peaks)  # should be == max_bands
    bands = []
    for i in range(len_peaks):
      if i == 0:
        range_px = (max(peaks[i] - (peaks[i + 1] - peaks[i]) / 2, 0),
                    (peaks[i + 1] - peaks[i]) / 2 + peaks[i])
      elif i == len_peaks - 1:
        range_px = ((peaks[i] - peaks[i - 1]) / 2 + peaks[i - 1],
                    min((peaks[i] - peaks[i - 1]) / 2 + peaks[i],imgh))
      else:
        range_px = ((peaks[i] - peaks[i - 1]) / 2 + peaks[i - 1],
                    (peaks[i + 1] - peaks[i]) / 2 + peaks[i])
      bands.append(self._one_band_item(peaks[i], range_px, range_hz, img))
    return bands

  def _one_band_item(self, center_px, range_px, range_hz, img):
    """pixels to hertz (top to bottom) + split image"""
    imgh = img.size[1]
    pxrg = (0, imgh)
    center_hz = utils.px2hz(imgh - center_px,   pxrg, range_hz)
    max_hz    = utils.px2hz(imgh - range_px[0], pxrg, range_hz)
    min_hz    = utils.px2hz(imgh - range_px[1], pxrg, range_hz)
    width_hz  = max_hz - min_hz
    bimg = img.crop((0, int(range_px[0]), img.size[0], int(range_px[1])))
    return (center_hz, width_hz, bimg)

  def _reduce_peaks(self, peaks, redux):
    """reduce the number of peaks"""
    data   = list(redux.getdata())
    pvals  = sorted([data[p] for p in peaks], reverse=True)
    thresh = pvals[self.conf['max_bands'] - 1]
    peaks  = [p for p in peaks if data[p] >= thresh]
    while len(peaks) > self.conf['max_bands']:
      peaks.remove([p for p in peaks if data[p] == thresh][-1])
    return peaks

  def _track_start(self, align, dur, max_dur, hues):
    """align track start time"""
    if   align == 'right':
      offset = max_dur - dur
    elif align == 'last_med_h':
      median = ImageStat.Stat(hues[-1]).median[0] / 255
      offset = (max_dur - dur) * median
    elif align == 'center':
      offset = (max_dur - dur) / 2
    elif align == 'first_med_h':
      median = ImageStat.Stat(hues[-1]).median[0] / 255
      offset = (max_dur - dur) * (1 - median)
    elif align == 'left':
      offset = 0
    return offset

  # generate sources ----------------------------------------------------------
  def _gen_sources(self, trackl):
    """generate source soundfiles"""
    log.info("generating sound sources")
    log.info("choices: %s", pformat_choices(self.song.choices))
    prog = 42
    for tid, tdata in trackl.iteritems():
      if tdata is not None:
        if tdata['img'] is not None:  # shared source image
          tdata['snd'] = self._gen_one_source(tid, tdata['img'],
                                              tdata['dur'], 'sh')
        else:
          for i, sect in zip(range(1, len(tdata['sections']) + 1),
                             tdata['sections']):
            sect['snd'] = self._gen_one_source(tid, sect['img'],
                                               sect['dur'],
                                               '%02d' % i)
      prog += 6
      self._save_runfile(prog)

  def _gen_one_source(self, tid, img_ref, dur, idpost):
    """generate one sound source"""
    fname = 'src-%s-%s.wav' % (tid, idpost)
    log.debug("   generating '%s' source (%s-%s, %7.3fs): %s",
              tid, img_ref[0], img_ref[1].upper(), dur, fname)
    if   tid == 'b':  # bustrof uses full size images, all channels
      img = self._bustrof_source_image(img_ref[0], 'hsl')
    elif tid == 'p':  # picspec uses one-channel reduxes
      chan_images = self.song.hsls[img_ref[1]]
      img = chan_images[self.song.img_idx.index(img_ref[0])]
    elif tid == 's':  # sinoide is also happy with the redux
      chan_images = self.song.hsls[img_ref[1]]
      img = chan_images[self.song.img_idx.index(img_ref[0])]
    gen = getattr(self, '_gen_sound_' + tid)
    gen(img, dur, fname)
    self._play_demo_sound(fname)
    return fname

  def _gen_sound_b(self, img, dur, fname):
    """
    generate a bustrof source sound
    uberoptimized by alecu@vortech.com.ar using numpy
    """
    # XXX to-do: cache <sdata>s
    fimg = img.convert('F')
    imga = numpy.fromstring(fimg.tobytes(), numpy.float32)
    srate = int(len(imga) / dur)
    log.debug("   bustrof source: %s (%s, %sx%s, sr: %s)",
              img.info['id'], img.mode, img.size[0], img.size[1], srate)
    imga = imga * 256
    imgu = imga.astype(numpy.uint16)
    iw, ih = img.size
    for desde in range(iw, len(imgu), iw*2):
        hasta = desde + iw
        imgu[desde:hasta] = numpy.array(imgu[hasta-1:desde-1:-1])
    sf = wave.open(fname, 'wb')
    sf.setparams((1, 2, srate, 0, 'NONE', 'not compressed'))
    sf.writeframesraw(imgu.tostring())
    sf.close()

  def _gen_sound_p(self, img, dur, fname):
    """generate a picspec source sound using enscribe"""
    no_ext    = os.path.splitext(fname)[0]
    img_fname = no_ext + '.jpg'
    ens_fname = no_ext + '.enscribe.wav'
    img.save(img_fname, 'JPEG')
    ens_tbr = random.choice(('-hiss', ''))
    shexec(self.conf['enscribe_cmd'], ens_tbr, img_fname, ens_fname)
    nsamples = wave.open(ens_fname, 'rb').getnframes()
    srate = int(ceil(nsamples / dur))
    amp = float(shexec('sox %s -n stat -v', ens_fname)) - 0.01
    log.debug("   sox amp: %7.3f", amp)
    shexec('sox -r %d %s %s vol %.3f', srate, ens_fname, fname, amp)

  def _gen_sound_s(self, img, dur, fname, norm=False):
    """generate a sinoide source sound"""
    # XXX to-do: cache <one_sec>s
    bands = self._image_split_bands(img, self.conf['sinoid_range'])
    freqs = [f for f, w, i in bands]
    if norm:
      amps  = [((48000 - f) / 48000) * (ImageStat.Stat(i).mean[0] / 256)
               for f, w, i in bands]
    else:
      amps  = [ImageStat.Stat(i).mean[0] / 255 for f, w, i in bands]
    log.debug("   generating sinoid for freqs: %s", zip(freqs, amps))
    rs = zip([2 * pi * freq / self.conf['sample_rate'] for freq in freqs],
             [amp * pow(8, 5) / len(amps) for amp in amps])
    vals = [sum([int(ar * sin(i * fr)) for fr, ar in rs])
            for i in range(self.conf['sample_rate'])]
    maxv = max(vals)
    if maxv != 0:
      norm = (pow(8, 5) - 16) / (maxv * 2)
    else:
      log.debug("   refusing to  normalize silent sfile!")
      norm = 1
    one_sec = ''
    for i in range(self.conf['sample_rate']):
      one_sec += utils.int2le16(int(vals[i] * norm))
    sf = wave.open(fname, 'wb')
    sf.setparams((1, 2, self.conf['sample_rate'], 0, 'NONE', 'not compressed'))
    sf.writeframesraw(one_sec * int(ceil(dur)))
    sf.close()

  def _bustrof_source_image(self, img_id, channel):
    """full size source image"""
    img = self.song.estim.get_image(img_id)
    while img.size[0] > 1024 or img.size[1] > 1024:
      img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.ANTIALIAS)
      log.warning("   image too big, resized to %s", img.size)
    if channel != 'hsl':
      log.debug("   splitting hsl bands from %s (%dx%d)",
                img_id, img.size[0], img.size[1])
      hsl = utils.image_hsl_split(img)
      img = hsl['hsl'.index(channel)]
    return img

  # generate tracks -----------------------------------------------------------
  def _filter_sections(self, trackl):
    """generate each track"""
    log.info("generating %d tracks", len(trackl))
    log.info("choices: %s", pformat_choices(self.song.choices))
    mixl = []
    prog = 60
    for tid, tdata in trackl.iteritems():
      log.debug("   generating track '%s' sections", tid)
      log.info("choices: %s", pformat_choices(self.song.choices))
      if tdata['snd'] is not None:
        self._split_shared_sound(tid, tdata['snd'], tdata['sections'])
      partl = []
      for i, sdata in zip(range(1, len(tdata['sections']) + 1),
                          tdata['sections']):
        if sdata['bands']:
          partl.append(self._filter_section(tid, sdata, i))
        prog += 10 / len(tdata['sections'])
        self._save_runfile(int(prog))
      mixl.append([tdata['start'], partl])
    self.song.mixlist = mixl
    log.debug("   saving mixlist ...")
    utils.save_bzdict('mixlist.pyd.bz2', mixl)
    return mixl

  def _split_shared_sound(self, tid, snd_src, sectl):
    """generate .ewf files"""
    len_sectl = len(sectl)
    log.debug("   splitting shared source in %d parts", len_sectl)
    for i, sdata in zip(range(1, len_sectl + 1), sectl):
      ewf_fname = 'src-%s-%02d.ewf' % (tid, i)
      ewf_text  = (("source = %s\n" +
                    "offset = 0.0\n" +
                    "start-position = %.5f\n" +
                    "length = %.5f\n" +
                    "looping = false\n") %
                   (snd_src, sdata['start'], sdata['dur']))
      open(ewf_fname, 'w').write(ewf_text)
      sdata['snd'] = ewf_fname

  def _filter_section(self, tid, sdata, i):
    """apply bpfs to source parts using ecasound"""
    log.debug("   applying filters to section %02d @ %s",
              i, [b['center'] for b in sdata['bands']])
    src_fname = sdata['snd']
    ftd_fname = 'ftd-%s-%02d.wav' % (tid, i)
    len_bands = len(sdata['bands'])
    amp_limit = 80 // len_bands
    rg_bands  = range(1, len_bands + 1)
    cmd = ("ecasound -f:16,2,%d -a:%s -i:resample-hq,auto,%s -o:%s"
           % (self.conf['sample_rate'], ','.join(str(b) for b in rg_bands),
              src_fname, ftd_fname))
    for j, bdata in zip(rg_bands, sdata['bands']):
      cmd += (" -a:%d -erc:1,2 -efs:%d,%d " %
              (j, bdata['center'], bdata['width']))
      lcmd = "-eac:50,1 -klg:1,0,100,%d" % len(bdata['amps'])
      rcmd = "-eac:50,2 -klg:1,0,100,%d" % len(bdata['amps'])
      if self.conf['normalize_amps']:
        all_amps   = ([d[1] for d in bdata['amps']] +
                      [d[2] for d in bdata['amps']])
        amax, amin = max(all_amps), min(all_amps)
        amp_base   = amin
        amp_ratio  = (amax - amin)
        if amp_ratio == 0:
          amp_base, amp_ratio = 0, 1
      else:
        amp_base, amp_ratio = 0, 1
      pt, pl, pr = -1, -1, -1
      last_amp = len(bdata['amps']) - 1
      for k, (t, l, r) in enumerate(bdata['amps']):
        if (t != pt and (l != pl or r != pr)) or k == last_amp:
          lcmd += ",%.2f,%.2f" % (t, (l - amp_base) / amp_ratio - 0.1)
          rcmd += ",%.2f,%.2f" % (t, (r - amp_base) / amp_ratio - 0.1)
          pt, pl, pr = t, l, r
      cmd += lcmd + ' ' + rcmd + (' -eal:%d' % amp_limit)
    shexec(cmd)
    self._play_demo_sound(ftd_fname)
    return (ftd_fname, sdata['dur'])

  def _filter_section_old(self, tid, sdata, i):
    """apply bpfs to source parts using ecasound"""
    log.debug("   applying filters to section %02d @ %s",
              i, [b['center'] for b in sdata['bands']])
    src_fname = sdata['snd']
    ftd_fname = 'ftd-%s-%02d.wav' % (tid, i)
    len_bands = len(sdata['bands'])
    amp_limit = 95 // len_bands
    rg_bands  = range(1, len_bands + 1)
    cmd = ("ecasound -f:16,2,%d -a:%s -i:resample-hq,auto,%s -o:%s"
           % (self.conf['sample_rate'], ','.join(str(b) for b in rg_bands),
              src_fname, ftd_fname))
    for j, bdata in zip(rg_bands, sdata['bands']):
      cmd += (" -a:%d -erc:1,2 -efs:%d,%d " %
              (j, bdata['center'], bdata['width']))
      lcmd = "-eac:50,1 -klg:1,0,100,%d" % len(bdata['amps'])
      rcmd = "-eac:50,2 -klg:1,0,100,%d" % len(bdata['amps'])
      if self.conf['normalize_amps']:
        all_amps   = ([d[1] for d in bdata['amps']] +
                      [d[2] for d in bdata['amps']])
        amax, amin = max(all_amps), min(all_amps)
        amp_base   = amin
        amp_ratio  = (amax - amin)
        if amp_ratio == 0:
          amp_base, amp_ratio = 0, 1
      else:
        amp_base, amp_ratio = 0, 1
      pt, pl, pr = -1, -1, -1
      last_amp = len(bdata['amps']) - 1
      for k, (t, l, r) in enumerate(bdata['amps']):
        if (t != pt and (l != pl or r != pr)) or k == last_amp:
          lcmd += ",%.2f,%.2f" % (t, (l - amp_base) / amp_ratio)
          rcmd += ",%.2f,%.2f" % (t, (r - amp_base) / amp_ratio)
          pt, pl, pr = t, l, r
      cmd += lcmd + ' ' + rcmd + (' -eal:%d' % amp_limit)
    shexec(cmd)
    self._play_demo_sound(ftd_fname)
    return (ftd_fname, sdata['dur'])

  # mix song ------------------------------------------------------------------
  def _mix_tracks(self, mixl):
    """mix all sections"""
    # XXX to-do: fix sgdur
    log.info("mixing %d tracks", len(mixl))
    log.info("choices: %s", pformat_choices(self.song.choices))
    share_path = os.path.join(oe3_path, 'share', 'spectrofoto')
    orc_path, sco_path = '%s.orc' % self.song.id, '%s.sco' % self.song.id
    csw_path, wav_path = '%s-cs.wav' % self.song.id, '%s.wav' % self.song.id
    shutil.copy(os.path.join(share_path, 'mix.orc' ), orc_path)
    sco   = open(os.path.join(share_path, 'mix.sco.pre'), 'r').read()
    envd  = {'long': (3, 0.25), 'mid': (2, 0.125), 'short': (1, 0.0625)}
    amp   = 1 / len(mixl)
    env, fade_amt = envd[self.song.choices['transitions'][0]]
    sgdur = 0
    for trackl in mixl:
      start, tmixl = trackl
      prev_end = (tmixl[0][1] * fade_amt) + start
      for i, (sfile, sdur) in enumerate(tmixl):
        fadet = sdur * fade_amt
        mom   = i == 0 and start or prev_end - fadet
        sco  += ('i1  %7.3f  %7.3f  "%s"  %5.3f   %d\n' %
                 (mom, sdur, sfile, amp, env))
        prev_end = mom + sdur
      sco += '\n'
      sgdur = max(prev_end, sgdur)
    open(sco_path, 'w').write(sco)
    cso = shexec("csound -dW -o %s %s %s", csw_path, orc_path, sco_path)
    maxes = re.search(r'overall amps: *([0-9.]+) *([0-9.]+)', cso).groups()
    amp = 30000 / max(float(x) for x in maxes)
    log.debug("   normalizing (amp: %.3f)", amp)
    shexec("sox -t .wav %s -t .wav %s vol %.3f", csw_path, wav_path, amp)
    os.unlink(csw_path)
    self.song.sfile = os.path.join(os.getcwd(), wav_path)

  # wrap song -----------------------------------------------------------------
  def _wrap_song(self):
    """move files, copy estim, save meta, remove temp dir"""
    # XXX clean up...
    log.info("wrapping song")
    sid  = self.song.id
    base = os.path.join(oe3_path, 'var/opus', sid)
    wpath, mpath = base + '.wav', base + '.pyd.bz2'
    # move wav
    log.debug("   moving wav to %s", wpath)
    os.rename(self.song.sfile, wpath)
    self.song.sfile = wpath
    # move tmp files
    log.debug("   creating tar archive: %s.tar", base)
    os.mkdir(base)
    for f in ('anal.pyd.bz2', 'choices.pyd.bz2', 'tracklist.pyd.bz2',
              'mixlist.pyd.bz2', sid + '.orc', sid + '.sco'):
      os.rename(f, os.path.join(base, f))
    os.chdir(os.path.join(oe3_path, 'var/opus'))
    tarfile.open(sid + '.tar', 'w').add(sid)
    # copy estim
    epath = os.path.join(base, self.song.estim.id + '.pyd.bz2')
    log.debug("   copying estim to %s", epath)
    self.song.estim.save(epath)
    # save metadata
    log.debug("   saving metadata to %s", mpath)
    wparams = wave.open(wpath, 'r').getparams()
    self.song.dur = wparams[3] / wparams[2]
    meta = dict((k, v) for k, v in self.song.__dict__.iteritems() if k in
                ('anal', 'choices', 'comp', 'dur', 'id', 'img_anomaly',
                 'img_distances', 'img_idx', 'img_medians', 'seed'))
    meta['sfile'] = sid + '.ogg'
    utils.save_bzdict(mpath, meta)
    # encode wav
    log.debug("   oggencoding ...")
    date = "%s-%s-%s %s:%s:%s" % (sid[:4], sid[4:6], sid[6:8],
                                  sid[9:11], sid[11:13], sid[13:15])
    shexec("oggenc -Q -b 256 -a 'oveja electrica' -d '%(date)s'" +
           " -t %(sid)s -c" +
           " license=http://creativecommons.org/licenses/by-sa/2.5/ar %(wav)s",
           date=date, sid=sid, wav=wpath)
    os.unlink(wpath)
    self.song.sfile= base + '.ogg'
    # clean up
    log.debug("   cleaning up ...")
    shutil.rmtree(sid)
    os.chdir(oe3_path)
    shutil.rmtree(self.song.tmpd)
    self._save_runfile(100)
    self._save_runfile(100, clear=True)


  # history stuff -------------------------------------------------------------
  """
  history:  [(<song_id>, <estim_id>, <analysis>, <choices>, <reward>), ...]
  analysis: {'num_images': <int>, 'red_min': <int>, ...}
  choices:  {'sort': ('median_l', <cert>), ...} where <cert> is an int
  reward:   (<reward>, <reward_detail>, <cert>)
  detail:   XXX to be defined, None for now.
  cert:     XXX to be defined, None for now.
  """

  def _aspect_history(self, aspect, option):
    """list of (<anal_value>, <choice>, <reward>) for aspect & option"""
    histo = self.state['histo']
    return [(anal[aspect], choices[option][0], reward[0])
            for sid, eid, anal, choices, reward in self.state['histo']]

  def _choice_history(self, option, vlist):
    """list of (<choices>, <reward>) where <option> in <vlist>"""
    histo = self.state['histo']
    return [(choices, reward[0])
            for sid, eid, anal, choices, reward in self.state['histo']
            if choices[option][0] in vlist]

  # other stuff ---------------------------------------------------------------
  def __init__(self):
    """load conf and state"""
    log.info("initializing new spectrofoto subcomposer, id: %s", id(self))
    log.debug("   loading conf from %s", conf_path)
    self.conf = utils.load_dict(conf_path)
    state_path = os.path.join(oe3_path, self.conf['state_path'])
    if os.path.isfile(state_path):
      log.debug("   loading state from %s", state_path)
      self.load_state(state_path)
    else:
      log.debug("   virgin composer. starting with a new state")
      self._init_state()

  def __del__(self):
    """save state"""
    state_path = os.path.join(oe3_path, self.conf['state_path'])
    self.save_state(state_path)

  def _init_state(self):
    """initialize subcomp state"""
    self.state = {'histo': []}

  def save_state(self, path=None):
    """save state to filesystem"""
    if path is None: path = os.path.join(oe3_path, self.conf['state_path'])
    utils.save_bzdict(path, self.state)

  def load_state(self, path=None):
    """load state from filesystem"""
    if path is None: path = os.path.join(oe3_path, self.conf['state_path'])
    self.state = utils.load_bzdict(path)

  def _save_runfile(self, prog, clear=False):
    """save comp state to runfile"""
    if clear:
      utils.save_runfile('comp', {});
      log.debug("   clearing comp runfile...")
      return  # shortcircuit
    #log.debug("   updating comp runfile...")
    dict_ = {k: v for k, v in self.song.__dict__.items() if k in
                ('anal', 'choices', 'comp', 'dur', 'id', 'img_anomaly',
                 'img_distances', 'img_idx', 'img_medians', 'seed')}
    dict_['estim_id']   = self.song.estim.id
    dict_['estim_name'] = self.song.estim.name
    dict_['images']     = [img.info['id'] for img in self.song.images]
    dict_['reduxes']    = {i.info['id']:i.info['redux']
                         for i in self.song.images}
    dict_['progress']   = prog
    utils.save_runfile('comp', dict_)
    if self.conf['sleep_on_update'] != 0:
      time.sleep(self.conf['sleep_on_update'])

  def _play_demo_sound(self, fname):
    """play a demo of one sound"""
    if self.conf['play_demos']:
      log.debug("   playing first bit of %s", fname)
      utils.play_sound(fname, 3.0)

  def _history_table(self):
    """generate a table view of the history"""
    akeys = ('red_min', 'red_max', 'red_delta', 'red_mean', 'red_stdev',
             'grn_min', 'grn_max', 'grn_delta', 'grn_mean', 'grn_stdev',
             'blu_min', 'blu_max', 'blu_delta', 'blu_mean', 'blu_stdev',
             'gry_min', 'gry_max', 'gry_delta', 'gry_mean', 'gry_stdev',
             'top_min', 'top_max', 'top_delta', 'top_mean', 'top_stdev',
             'mid_min', 'mid_max', 'mid_delta', 'mid_mean', 'mid_stdev',
             'btm_min', 'btm_max', 'btm_delta', 'btm_mean', 'btm_stdev',
             'dst_min', 'dst_max', 'dst_delta', 'dst_mean', 'dst_stdev')
    ckeys = ('sort', 'randomize', 'transform', 'sect_base_time',
             'sect_dur_ratio', 'sect_align', 'env_follow', 'transitions',
             'timbres', 'spectrum_source', 'filter')
    tbl = (
        "       -----blue----- ----bottom---- ----distn1---- " +
        "----green----- -----gray----- ----middle---- -----red------ "+
        "-----top------  ---estim--- ------section------ ---timbre--\n" +
        " n  ni mn mx dt md sd mn mx dt md sd mn mx dt md sd " +
        "mn mx dt md sd mn mx dt md sd mn mx dt md sd mn mx dt md sd " +
        "mn mx dt md sd  " +
        "sor rnd tfm sbt sdr sal env tns tbr src flt  rw\n" +
        "--  ------------------------------------------------" +
        "------------------------------------------------------------"+
        "--------------  " +
        "-------------------------------------------  --\n")
    for i, row in enumerate(self.state['histo']):
      anal = ' '.join(['%2s' % str(row[2][key])[-2:] for key in akeys])
      choices = ' '.join(['%3s' % str(row[3][key][0])[:3]
                          for key in ckeys])
      tbl += '%2d  %2d %s  %s  %+d\n' % (i, row[2]['num_images'], anal,
                                      choices, row[4][0])
    return tbl

def pformat_choices(choices, joined=True):
  """format choices for printing"""

  ckeys = (('sort', 'srt'), ('randomize', 'rnd'), ('transform', 'tfm'),
           ('sect_base_time', 'sbt'), ('sect_dur_ratio', 'sdr'),
           ('sect_align', 'sal'), ('env_follow', 'efl'),
           ('transitions', 'tns'), ('timbres', 'tbr'),
           ('spectrum_source', 'src'), ('filter', 'flt'))
  timbres = {'s': u'\xb7\xb7s', 'p': u'\xb7p\xb7', 'ps': u'\xb7ps',
             'b': u'b\xb7\xb7', 'bs': u'b\xb7s', 'bp': u'bp\xb7', 'bps': 'bps'}

  # XXX esto es hoyyible
  choice_l = ('%s:%3s.%02d' % (v,
                               (isinstance(choices[k][0], int)
                                and '%03d' % choices[k][0]
                                or (choices[k][0][:4] == 'inv_'
                                    and (choices[k][0][:2] +
                                         choices[k][0][4]))
                                or (v == 'tbr' and timbres[choices[k][0]])
                                or choices[k][0][:3]),
                               choices[k][1])
              for k, v in ckeys)
  return ' '.join(choice_l) if joined else choice_l

#------------------------------------------------------------------------------
if __name__ == '__main__':
  from oe3.utils import run_doctest
  run_doctest('comp_spectrofoto')
