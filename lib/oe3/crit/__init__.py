# oe3 lib/oe3/crit/__init__.py
#
# oveja electrica - oe3.0.1:bleu
# copyright (c) 2003-2017 santiago pereson <yaco@yaco.net>
# http://yaco.net/oe/
#
# this file is part of 'oe3', which is released under the gnu general public
# license, version 3. see LICENSE for full license details.
#

"""oe3 critic subsystem"""


from __future__ import division


__version__ = '0.0.1'
__all__ = ('REWARDS', 'MAX_REWARD')


REWARDS = (-3, -1, 0, +3, +7)
MAX_REWARD = max(REWARDS)
