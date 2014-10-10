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
import eval
import StringIO
import numpy as np

from ftex import FeatureTemplate


class AccuracyResults(dict):
    """POS tagger accuracy results container class.
    """

    _total_name = 'Total'

    @property
    def total(self):
        """Name of total accuracy key in the results dictionary.


        :return: total results key
        :rtype: str
        """
        return self._total_name

    @total.setter
    def total(self, name):
        self._total_name = name

    def export_to_file(self, fp, *args, **kwargs):
        """Export results to a file.

        :param fp: file path
        :type fp: str
        """
        with open(fp, 'w') as f:
            f.write('-----------------------\n')
            for k in kwargs.keys():
                if type(kwargs[k]) is not list:
                    kwargs[k] = [kwargs[k]]
                f.write('%s: %s' % (k, ' '.join([str(x) for x in kwargs[k]])))
                f.write('\n-----------------------\n')
            for i in self.items():
                if i[0] == 'Total':
                    continue
                k = i[0]
                n_cor, n_all, acc = i[1]
                f.write('%s: %s (%s/%s)\n' % (k, acc, n_cor, n_all))
            f.write('-----------------------\n')
            f.write('%s: %s (%s/%s)\n' % (self.total,
                                          self[self.total][2],
                                          self[self.total][0],
                                          self[self.total][1]))
            f.write('-----------------------\n')

    def __setitem__(self, key, value):
        tuple_val = value if type(value) is tuple else (0, 0, value)
        super(AccuracyResults, self).__setitem__(key, tuple_val)

    def __str__(self):
        rf = StringIO.StringIO()
        rf.write('-----------------------\n')
        for i in self.items():
            if i[0] == 'Total':
                continue
            k = i[0]
            n_cor, n_all, acc = i[1]
            rf.write('%s: %s (%s/%s)\n' % (k, acc, n_cor, n_all))
        rf.write('-----------------------\n')
        rf.write('%s: %s (%s/%s)\n' % (self.total,
                                       self[self.total][2],
                                       self[self.total][0],
                                       self[self.total][1]))
        rf.write('-----------------------\n')
        return rf.getvalue()

    def __repr__(self):
        return self.__str__()


def parse_ftvec_templ(s, r):
    ftt = FeatureTemplate()
    fts_str = [x for x in s.strip().replace(' ', '').split(';')]
    for ft in fts_str:

        # empty featues (...; ;feature:params)
        if ft.strip() == '':
            continue

        # no parameter features
        if ':' not in ft:
            ftt.add_feature(ft, (ft,))
            continue
        elif ft.count(':') == 1 and ft.endswith(':'):
            ftt.add_feature(ft[:-1], (ft[:-1],))
            continue

        # function name & parameter values
        fn, v = ft.split(':', 1)

        # value matches
        m = re.match('(?:\[([0-9:,-]+)\])?(.+)?', v)

        # window range
        fw = parse_range(m.group(1)) if m.group(1) else None

        # function parameters
        fp = ()

        # adding resources if necessary
        if fn in r.keys():
            fp += tuple(r[fn])

        # adding function parameters if specified
        if m.group(2) is not None:
            fp += tuple(x for x in m.group(2).split(',') if x)

        # name, window, parameters
        ftt.add_win_features(fn, fw, fp)

    return ftt


def parse_range(r):
    """Parses a range in string representation adhering to the following format:
    1:3,6,8:9 -> 1,2,3,6,8,9

    :param r: range string
    :type r: str
    """
    rng = []

    # Range strings
    rss = [x.strip() for x in r.split(',')]

    for rs in rss:
        if ':' in rs:
            # Range start and end
            s, e = (int(x.strip()) for x in rs.split(':'))
            for i in range(s, e + 1):
                rng.append(int(i))
        else:
            rng.append(int(rs))

    return rng


def parse_conll_eval_table(fp):
    table = {}

    with open(fp, 'r') as tbl:
        tbl.readline()
        for row in tbl:
            cells = [
                x.strip().replace('%', '').replace('\\', '')
                for x
                in row[:-9].split("&")
            ]
            table[cells[0]] = (cells[1], cells[2], cells[3])

    return table


def parse_data(fp, cols, ts='\t'):
    ct = {'pos': ('form', 'postag'), 'chunk': ('form', 'postag', 'chunktag')}
    c = ct[cols] if type(cols) is str else cols
    rc = count_records(fp)
    nc = len(c)
    dt = 'a60,{},int32'.format(','.join('a10' for _ in range(nc)))
    data = np.zeros(rc, dtype=dt)
    names = c + ('guesstag', 'eos')
    data.dtype.names = names
    with open(fp, 'r') as fh:
        idx = 0
        start = 0
        for line in fh:
            if line.strip() == '':
                data[start]['eos'] = idx
                start = idx
                continue
            data[idx] = tuple(line.strip().split(ts)[:len(c)]) + ('', -1)
            idx += 1
    return data


def count_records(fp):
    c = 0
    with open(fp, 'r') as fh:
        for l in fh:
            if l.strip() != '':
                c += 1
    return c


class SequenceData:

    def __init__(self, data):
        """

        :param data: sequence data
        :type data: np.array
        """
        self.data = data

    @property
    def sequences(self):
        return list(self.gsequences)

    def gsequences(self, cols=None):

        c = list(self.data.dtype.names) if cols is None else cols

        # sequence start and end indices
        s, e = 0, 0

        # extracting features from sequences
        while 0 <= s < len(self.data):

            # index of the end of a sequence is recorded at the beginning
            e = self.data[s]['eos']

            # slicing a sequence
            seq = self.data[s:e]

            # moving the start index
            s = e

            # returning a sequence
            yield seq[c]

    def export_to_file(self, fp, cols=None, ts='\t'):
        c = list(self.data.dtype.names) if cols is None else cols
        with open(fp, 'w') as fh:
            d = self.data[c]
            for i in xrange(len(self.data)):
                fh.write(ts.join(str(x) for x in d[i]))
                fh.write('\n')

    @property
    def accuracy(self):
        """Returns the POS tag accuracy based on the `postag` and `guesstag`
        fields in the contained record array object.

        :return: guess accuracy results by category
        :rtype: AccuracyResults
        """
        cc = {}
        ac = {}
        for it in self.data:
            if it['postag'] not in ac.keys():
                ac[it['postag']] = 0.0
                cc[it['postag']] = 0.0
            if it['postag'] == it['guesstag']:
                cc[it['postag']] += 1
            ac[it['postag']] += 1

        tcc = 0.0
        tac = 0.0

        results = eval.AccuracyResults()

        for t in ac.keys():
            results[t] = (cc[t], ac[t], cc[t] / ac[t])
            tcc += cc[t]
            tac += ac[t]

        results['Total'] = (tcc, tac, tcc / tac)

        return results

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __iter__(self):
        return self.data.__iter__()

    def __reversed__(self):
        reversed(self.data)
        return self