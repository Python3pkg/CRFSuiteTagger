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

import os
import re
import sys
import time
import StringIO
import traceback
import random as rnd

from os.path import join
from utils import export
from iterpipes import check_call, cmd
from subprocess import CalledProcessError


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
        return self[self._total_name]

    def parse_conll_eval_table(self, fp):
        """Parses the LaTeX table output of the CoNLL-2000 evaluation script
        into this object.

        :param fp: file path
        :type fp: str
        :return: results by category
        :rtype: dict
        """
        with open(fp, 'r') as tbl:
            tbl.readline()
            for row in tbl:
                clean_row = re.sub('([\\\\%]|hline)', '', row)
                cells = [x.strip() for x in clean_row.split('&')]
                self[cells[0]] = {
                    'precision': float(cells[1]),
                    'recall': float(cells[2]),
                    'fscore': float(cells[3])
                }
        self[self._total_name] = self['Overall']
        del self['Overall']

        return None

    def export_to_file(self, fp, *args, **kwargs):
        """Export results to a file.

        :param fp: file path
        :type fp: str
        """
        with open(fp, 'w') as fh:
            self._to_str(fh)

    def _pack_str(self, key):
        itm = self[key]
        return '%s ==> pre: %s, rec: %s, f: %s acc: %s\n' % (
            key,
            itm.get('precision', 'n.a.'),
            itm.get('recall', 'n.a.'),
            itm.get('fscore', 'n.a.'),
            itm.get('accuracy', 'n.a.')
        )

    def _to_str(self, fh):
        fh.write('--------------------------------------------------------\n')
        for k in self.keys():
            if k == self._total_name:
                continue
            fh.write(self._pack_str(k))
        fh.write('--------------------------------------------------------\n')
        fh.write(self._pack_str(self._total_name))
        fh.write('--------------------------------------------------------\n')

    def __str__(self):
        rf = StringIO.StringIO()
        self._to_str(rf)
        return rf.getvalue()

    def __repr__(self):
        return self.__str__()


def conll(data):
    """Evaluates chunking f1-score provided with data with the following fields:
    form, postag, chunktag, guesstag

    Currently uses the CoNLL-2000 evaluation script to make the estimate.

    :param data: np.array
    :return: f1-score estimate
    :rtype: AccuracyResults
    """
    # TODO: reimplement the conll testing script in Python
    try:
        os.makedirs(join(os.getcwd(), 'tmp/'))
    except OSError:
        pass

    td = join(os.getcwd(), 'tmp/')

    rn = rnd.randint(1000, 1000000000000)

    fp_dp = join(td, 'chdata.%s.%s.tmp' % (time.asctime().replace(' ', ''), rn))
    fp_res = join(td, 'chres.%s.%s.tmp' % (time.asctime().replace(' ', ''), rn))
    fh_out = open(fp_res, 'w')

    print '%s exporting' % time.asctime()
    export(data,
           open(fp_dp, 'w'),
           cols=['form', 'postag', 'chunktag', 'guesstag'],
           ts=' ')

    cwd = join(os.getcwd(), "prl/")
    c = cmd(
        'perl conll_eval.pl -l < {}',
        fp_dp,
        cwd=cwd,
        stdout=fh_out
    )

    r = AccuracyResults()

    try:
        check_call(c)
        r.parse_conll_eval_table(fp_res)
    except CalledProcessError:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print "*** print_tb:"
        traceback.print_tb(exc_traceback,
                           limit=1,
                           file=sys.stdout)
        print "*** print_exception:"
        traceback.print_exception(exc_type,
                                  exc_value,
                                  exc_traceback,
                                  limit=2,
                                  file=sys.stdout)
    finally:
        os.remove(fp_dp)
        os.remove(fp_res)
        return r


def pos(data):
    """Estimates POS tagging accuracy based on the `postag` and `guesstag`
    fields in `data`.

    :return: guess accuracy results by category
    :rtype: AccuracyResults
    """
    cc = {}  # correct count
    ac = {}  # all count
    for it in data:
        if it['postag'] not in ac.keys():
            ac[it['postag']] = 0.0
            cc[it['postag']] = 0.0
        if it['postag'] == it['guesstag']:
            cc[it['postag']] += 1
        ac[it['postag']] += 1

    tcc = 0.0  # total correct count
    tac = 0.0  # total all count

    results = AccuracyResults()

    for t in ac.keys():
        results[t] = {'accuracy': cc[t] / ac[t]}
        tcc += cc[t]
        tac += ac[t]

    results['Total'] = {'accuracy': tcc / tac}

    return results