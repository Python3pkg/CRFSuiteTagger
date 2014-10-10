# This file is part of CRFSuiteTagger.
#
# CRFSuiteTagger is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CRFSuiteTagger is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CRFSuiteTagger.  If not, see <http://www.gnu.org/licenses/>.
__author__ = 'Aleksandar Savkov'

import re
import numpy as np
from collections import OrderedDict


class FeatureContainer:
    """This class extends the `SSVList` class. Its main purpose is to store
    feature vectors generated from the items in a SSVList-like data list object
    provided to the constructor.

    """

    def __init__(self, data):
        """Constructs an object of this class.

        :param data: list of data points
        """
        self.data = data
        self.ftt = None
        self.fts = None

    def extract_features(self, ftt):
        """Populates this object with features stored in SSVRecord objects,
        generated using a feature extraction template.

        :param ftt: feature extraction template
        :type ftt: FeatureTemplate
        :param tab_sep: tab separator
        :type tab_sep: str
        """

        # feature extraction template
        self.ftt = ftt

        # number of features
        nft = len(ftt.dict.keys())

        # constructing empty features array
        rc = len(self.data)
        dt = 'a60,{}'.format(','.join('a30' for _ in range(nft)))
        self.fts = np.zeros(rc, dtype=dt)

        # sequence start, end indices
        start, end = 0, 0

        import time
        print '%s Processed sequences:' % time.asctime()

        sc = 0

        # extracting features from sequences
        while 0 < start < len(self.data):

            # index of the end of a sequence is recorded at the beginning
            end = self.data[start]['eos']

            # slicing a sequence
            seq = self.data[start:end]

            # moving the start index
            start = end

            # extracting the features
            for i in range(len(seq)):
                self.fts[i] = tuple(self.ftt.make_fts(seq, i))

            sc += 1

            if sc % 1000 == 0:
                print '%s' % sc

    @property
    def sequences(self):
        return list(self.gsequences)

    @property
    def gsequences(self):

        # sequence start and end indices
        s, e = 0, 0

        # extracting features from sequences
        while 0 <= s < len(self.data):

            # index of the end of a sequence is recorded at the beginning
            e = self.data[s]['eos']

            # slicing a sequence
            seq = self.fts[s:e]

            # moving the start index
            s = e

            # returning a sequence
            yield seq

    def __len__(self):
        return len(self.fts)

    def __getitem__(self, item):
        return self.fts[item]

    def __setitem__(self, key, value):
        self.fts[key] = value

    def __iter__(self):
        return self.fts.__iter__()

    def __reversed__(self):
        reversed(self.fts)
        return self


class FeatureTemplate:

    def __init__(self, t=None, spec_ft=None):
        self.dict = OrderedDict() if t is None else t
        self.sfts = ['emb'] + spec_ft if spec_ft else ['emb']
        self.resources = {}

    def make_fts(self,
                 data,
                 i,
                 form_name='form',
                 *args, **kwargs):
        ret = [data[i][form_name]]
        for k in self.dict.keys():
            f = self.dict[k][0]
            p = self.dict[k][1:]
            func = FeatureTemplate.__dict__[f].__func__ if type(f) is str else f
            ret.append(func(data, i, *(p + args), **kwargs))
        return ret

    def add_feature(self, n, ps):
        """

        :param n: feature name
        :param ps: feature function parameters
        """
        self.dict[n] = ps

    def add_win_features(self, fn, fw, fp, *args, **kwargs):
        """

        :param fn: function name
        :param params: feature function parameters
        """
        if type(fn) is str:
            if fn in self.sfts:
                func = getattr(self, '%s_win' % fn)
            else:
                func = self.generic_win
        else:
            func = fn
        for n, v in func(fn, fw, fp, *args, **kwargs):
            self.dict[n] = v

    @staticmethod
    def generic_win(fn, fw, fp, np='%s%s', *args, **kwargs):
        n, f = (fn, fn) if type(fn) is str else (fn.__name__, fn)
        prms = tuple() if fp is None else tuple(fp)
        for i in fw:
            yield np % (n, str(i)), (f, i) + prms

    @staticmethod
    def emb_win(fn, fw, fp, np='emb%s:%s', *args, **kwargs):
        """

        :param fn: function name
        :param fw: embeddings window (range of ints)
        :param fp: embeddings
        """
        e, = fp
        emb_vec_size = len(e[e.keys()[0]])
        for i in fw:
            for j in range(emb_vec_size):
                yield np % (i, j), (fn, i, j, e)


    @staticmethod
    def word(data, i, rel=0, *args, **kwargs):
        try:
            form = data[i + rel]['form']
        except (IndexError, KeyError):
            form = None
        return 'w[%s]=%s' % (i + rel, form)

    @staticmethod
    def pos(data, i, rel=0, *args, **kwargs):
        try:
            postag = data[i + rel]['postag']
        except (IndexError, KeyError):
            postag = None
        return 'pos[%s]=%s' % (i + rel, postag)

    @staticmethod
    def chunk(data, i, rel=0, *args, **kwargs):
        try:
            chunktag = data[i + rel].chunktag
        except (IndexError, KeyError):
            chunktag = None
        return 'chunk[%s]=%s' % (i + rel, chunktag)

    @staticmethod
    def can(data, i, rel=0, *args, **kwargs):
        try:
            w = data[i + rel]['form']
            w = re.sub('\d', '#', w)
            w = re.sub('\w', 'x', w)
            w = re.sub('[^#x]', '*', w)
        except (IndexError, KeyError):
            w = None
        return 'can[%s]=%s' % (i + rel, w)

    @staticmethod
    def brown(data, i, rel=0, b=None, p=None, *args, **kwargs):
        """

        :param data: data
        :type data: DataFrame
        :param i: index
        :type i: int
        :param b: brown clusters
        :type b: dict
        :param rel: relative index
        :type rel: int
        :param p: prefix
        :type p: int
        :return: feature string
        :rtype str:
        """
        pref = 'full'
        try:
            cname = b[data[i + rel]['form']]
            if p:
                cname = cname[:p]
                pref = p
        except (KeyError, IndexError):
            cname = None
        return 'cname[%s]:%s=%s' % (rel, pref, cname)

    @staticmethod
    def cls(data, i, rel=0, c=None, *args, **kwargs):
        """

        :param data: data
        :type data: DataFrame
        :param i: index
        :type i: int
        :param c: clusters
        :type c: dict
        :param rel: relative index
        :type rel: int
        :return: feature string
        :rtype: str
        """
        try:
            cnum = c[data[i + rel]['form']]
        except (KeyError, IndexError):
            cnum = None
        return 'cnum[%s]=%s' % (rel, cnum)

    @staticmethod
    def emb(data, i, rel=0, j=0, e=None, *args, **kwargs):
        try:
            emb = e[data[i + rel]['form']][j]
        except (KeyError, IndexError) as ex:
            emb = None
        return 'emb[%s][%s]=%s' % (rel, j, emb)

    @staticmethod
    def isnum(data, i, rel=0, *args, **kwargs):
        try:
            isnum = bool(re.match('[0-9/]+', data[i + rel]['form']))
        except (IndexError, KeyError):
            isnum = None
        return 'isnum[%s]=%s' % (str(rel), isnum)