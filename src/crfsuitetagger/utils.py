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

import numpy as np


def parse_conll_eval_table(fp):
    """Parses the LaTeX table output of the CoNLL-2000 evaluation script into a
    dictionary with string keys and tuple values containing precision, recall,
    and f1-score values at the respective positions.

    :param fp: file path
    :type fp: str
    :return: results by category
    :rtype: dict
    """
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


def parse_tsv(fp, cols, ts='\t'):
    """Parses a file of TSV sequences separated by an empty line and produces
    a numpy recarray. The `cols` parameter can use a predefined set of field
    names or it can be user specific. The fields may be arbitrary in case new
    features/extractor functions are defined, however, a convention should be
    followed for the use of the POS tagging and chunking features included in
    this library.

    Example of a file with chunk data:

    <form>  <postag>    <chunktag>  <guesstag>

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


def export(data, fp, cols=None, ts='\t'):
    """ Exports recarray to a TSV sequence file, where sequences are divided by
    empty lines.

    :param data: data
    :type data: np.array
    :param fp: file path
    :type fp: str
    :param cols: column names
    :type cols: tuple or list
    :param ts:
    """
    c = list(data.dtype.names) if cols is None else cols
    with open(fp, 'w') as fh:
        d = data[c]
        for i in xrange(len(data)):
            fh.write(ts.join(str(x) for x in d[i]))
            fh.write('\n')


def gsequences(data, cols=None):
    """Returns a generator that yields a sequence from the provided data.
    Sequences are determined based on the `eos` field in `data`. If no column
    names are provided, all fields are included.

    :param data: data
    :type data: np.array
    :param cols: column names
    :type cols: list
    """
    c = list(data.dtype.names) if cols is None else cols

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