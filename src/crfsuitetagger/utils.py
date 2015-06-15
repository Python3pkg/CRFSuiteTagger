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
import copy
import os.path
import StringIO
import random
import string
import ConfigParser
import numpy as np


def parse_tsv(fp=None, cols=None, ts='\t', s=None):
    """Parses a file of TSV sequences separated by an empty line and produces
    a numpy recarray. The `cols` parameter can use a predefined set of field
    names or it can be user specific. The fields may be arbitrary in case new
    features/extractor functions are defined, however, a convention should be
    followed for the use of the POS tagging and chunking features included in
    this library.

    Example of a file with chunk data:

    <form>  <postag>    <chunktag>  <guesstag>

    Note: all configurations should start with <form>

    :param fp: file path
    :type fp: str
    :param cols: column names
    :type cols: str or tuple
    :param ts: tab separator
    :type ts: str
    :param s: TSV string
    :type s: str
    :return: parsed data
    :rtype: np.array
    """

    ct = {
        'pos': ('form', 'postag'),
        'chunk': ('form', 'postag', 'chunktag'),
        'ne': ('form', 'postag', 'chunktag', 'netag')
    }
    c = ct[cols] if type(cols) is str else cols

    if s is not None:
        stream = StringIO.StringIO()
        stream.write(s)
        stream.seek(0)
    elif fp is not None:
        stream = open(fp, 'r')
    else:
        raise ValueError('fp and s values are None. At least one of them must '
                         'be initialised.')

    rc = count_records(stream)
    nc = len(c) - 1
    cs = ','.join('a10' for _ in range(nc))  # col strings
    dt = 'a60,{}a10,int32'.format('%s,' % cs if cs else '')

    data = np.zeros(rc, dtype=dt)

    names = c + ('guesstag', 'eos')
    data.dtype.names = names

    idx = 0
    start = 0
    for line in stream.readlines():
        if line.strip() == '':
            data[start]['eos'] = idx
            start = idx
            continue
        # Note: len(c) is there to handle input data with more columns than
        # declared in the `cols` parameter.
        line_cols = line.strip().split(ts)
        empty_cols = ()
        if len(names) > len(c) + 1:
            empty_cols = \
                tuple('' for _ in range(len(names) - len(c) - 1))
        data[idx] = tuple(line_cols[:len(c)]) + empty_cols + (-1,)
        idx += 1
    if start < len(data):
        data[start]['eos'] = idx
    return data


def count_records(f):
    """Counts the number of empty lines in a file.

    :param f: file path
    :type f: FileIO or StringIO
    :return: number of empty lines
    :rtype: int
    """
    c = 0
    for l in f.readlines():
        if l.strip() != '':
            c += 1
    f.seek(0)
    return c


def export(data, f, cols=None, ts='\t'):
    """ Exports recarray to a TSV sequence file, where sequences are divided by
    empty lines.

    :param data: data
    :type data: np.array
    :param f: output stream
    :type f: FileIO or StringIO.StringIO
    :param cols: column names
    :type cols: list or str
    :param ts:
    """

    # column templates
    ct = {
        'pos': ['form', 'postag', 'guesstag'],
        'chunk': ['form', 'postag', 'chunktag', 'guesstag'],
        'ne': ['form', 'postag', 'chunktag', 'netag', 'guesstag']
    }

    # all columns in the data
    dt = data.dtype.names

    # columns to be exported
    c = (list(dt) if cols is None
         else ct[cols] if type(cols) is str else list(cols))

    rc = len(data)
    d = data[c]
    eos = None
    for i in xrange(rc):
        # index of the beginning of the next sequence
        eos = data[i]['eos'] if data[i]['eos'] > 0 else eos

        # writing current entry
        f.write(ts.join(str(x) for x in d[i]))

        # not writing a newline after last entry
        if i != rc - 1:
            f.write('\n')

        # writing an empty line after sequence
        if eos == i + 1:
            f.write('\n')


def gsequences(data, cols=None):
    """Returns a generator that yields a sequence from the provided data.
    Sequences are determined based on the `eos` field in `data`. If no column
    names are provided, all fields are included.

    :param data: data
    :type data: np.array
    :param cols: column names
    :type cols: list or str
    """
    # column templates
    ct = {'pos': ['form', 'postag', 'guesstag'],
                 'chunk': ['form', 'postag', 'chunktag', 'guesstag']}

    # all columns in the data
    dt = data.dtype.names

    # columns to be exported
    c = list(dt) if cols is None else ct[cols] if type(cols) is str else cols

    # sequence start and end indices
    s, e = 0, 0

    # extracting features from sequences
    while 0 <= s < len(data):

        # index of the end of a sequence is recorded at the beginning
        e = data[s]['eos']

        # slicing a sequence
        seq = data[s:e]

        # moving the start index
        s = e

        # returning a sequence
        yield seq[c]


def count_sequences(data):
    """Counts the number of sequences in the data.

    :param data: data
    :type data: np.array
    :return: number of sequences
    :rtype: int
    """
    # sequence count
    c = 0

    # sequence start and end indices
    s, e = 0, 0

    # extracting features from sequences
    while 0 <= s < len(data):

        # index of the end of a sequence is recorded at the beginning
        e = data[s]['eos']

        # moving the start index
        s = e

        # counting up
        c += 1

    return c


def set_sequence_start_idx(data, idx):
    """Sets all sentence indices relative to a starting value provided in `idx`.

    :param data: data
    :type data: np.array
    :param idx: relative starting index
    :type idx: int
    """

    assert idx >= 0, 'Negative indices are not supported.'

    # get the start of the second sentence (first eos index)
    eos_idx = 1
    while eos_idx < len(data) and data[eos_idx]['eos'] < 0:
        eos_idx += 1

    # index difference
    diff = idx + eos_idx - data[0]['eos']

    for i in range(len(data)):
        if data[i]['eos'] > 0:
            data[i]['eos'] += diff


def weighed_split(data, proportion=0.9):
    """Splits the data into two given a proportion.

    :param data: data
    :type data: np.array
    :param proportion: split proportion
    :type proportion: float
    :return: data_split1, data_split2
    :rtype: np.array, np.array
    """

    # sequence start and end indices
    s, e = 0, 0

    # ceiling
    ceil = int(proportion * len(data))

    # last sentence
    sh = 0

    # extracting features from sequences
    while 0 <= s < ceil:

        # index of the end of a sequence is recorded at the beginning
        e = data[s]['eos']

        # keeping history
        sh = s

        # moving the start index
        s = e

    data_1 = data[:sh]
    data_2 = data[sh:]

    set_sequence_start_idx(data_2, 0)

    return data_1, data_2


def cv_splits(data, k=10):
    """ Yields `k` cross-validation splits of `data`.

    :param data: data
    :type data: np.array
    :param k: folds
    :type k: int
    """

    assert k > 2, 'Folds value k too small (%s). Should be at least 3.' % k

    # we need a copy of the data
    d = copy.deepcopy(data)
    # calculate proportion
    prop = (float(k) - 1) / float(k)

    # yield splits and remake the data
    for _ in range(k):
        # split data
        trd, ted = weighed_split(d, proportion=prop)
        # make copies
        d1 = copy.deepcopy(trd)
        d2 = copy.deepcopy(ted)
        # make new data with small split in the beginning
        # big split indices need to start after small split
        set_sequence_start_idx(d1, idx=len(d2))
        # concatenate the data
        d = np.concatenate((d2, d1))

        yield trd, ted


def expandpaths(cfg):
    for sec in cfg.sections():
        for opt in cfg.options(sec):
            # option value
            ov = cfg.get(sec, opt)

            # does it look like it needs expanding
            if ov and re.match('^~/(?:[^\/]+/)*(?:[^\/]+)?', ov):
                cfg.set(sec, opt, os.path.expanduser(ov))


def clipcfg(cfg):
    c = copycfg(cfg)
    c.set('tagger', 'train', None)
    c.set('tagger', 'test', None)
    c.set('tagger', 'model', None)
    for o in c.options('resources'):
        c.set('resources', o, None)
    return c


def copycfg(cfg):
    c = ConfigParser.ConfigParser()
    buff = StringIO.StringIO()
    cfg.write(buff)
    buff.seek(0)
    c.readfp(buff)
    return c


def random_str(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))