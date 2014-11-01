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
import os.path
import numpy as np


def parse_tsv(fp, cols, ts='\t'):
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
    :return: parsed data
    :rtype: np.array
    """

    ct = {'pos': ('form', 'postag'), 'chunk': ('form', 'postag', 'chunktag')}
    c = ct[cols] if type(cols) is str else cols

    rc = count_records(fp)
    nc = len(c) - 1
    dt = 'a60,{},a10,int32'.format(','.join('a10' for _ in range(nc)))

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
            # Note: len(c) is there to handle input data with more columns than
            # declared in the `cols` parameter.
            data[idx] = tuple(line.strip().split(ts)[:len(c)]) + ('', -1)
            idx += 1
        if start < len(data):
            data[start]['eos'] = idx
    return data


def count_records(fp):
    """Counts the number of empty lines in a file.

    :param fp: file path
    :type fp: str
    :return: number of empty lines
    :rtype: int
    """
    c = 0
    with open(fp, 'r') as fh:
        for l in fh:
            if l.strip() != '':
                c += 1
    return c


def export(data, f, cols=None, ts='\t'):
    """ Exports recarray to a TSV sequence file, where sequences are divided by
    empty lines.

    :param data: data
    :type data: np.array
    :param f: output file
    :type f: FileIO or StringIO
    :param cols: column names
    :type cols: list or str
    :param ts:
    """

    # column templates
    ct = {'pos': ['form', 'postag', 'guesstag'],
                 'chunk': ['form', 'postag', 'chunktag', 'guesstag']}

    # all columns in the data
    dt = data.dtype.names

    # columns to be exported
    c = list(dt) if cols is None else ct[cols] if type(cols) is str else cols

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


def expandpaths(cfg):
    for sec in cfg.sections():
        for opt in cfg.options(sec):
            # option value
            ov = cfg.get(sec, opt)

            # does it look like it needs expanding
            if re.match('^~/(?:[^\/]+/)*(?:[^\/]+)?', ov):
                cfg.set(sec, opt, os.path.expanduser(ov))