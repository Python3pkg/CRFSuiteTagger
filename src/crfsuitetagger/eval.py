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
import sys
import time
import StringIO
import traceback

from os.path import join
from utils import export
from iterpipes import check_call, cmd
from utils import parse_conll_eval_table
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

    fp_dp = join(td, 'chdata.%s.tmp' % time.asctime().replace(' ', ''))
    fp_res = join(td, 'chres.%s.tmp' % time.asctime().replace(' ', ''))
    fh_out = open(fp_res, 'w')

    print '%s exporting' % time.asctime()
    export(data,
           open(fp_dp, 'w'),
           ['form', 'postag', 'chunktag', 'guesstag'],
           ' ')

    cwd = join(os.getcwd(), "prl/")
    c = cmd(
        'perl conll_eval.pl -l < {}',
        fp_dp,
        cwd=cwd,
        stdout=fh_out
    )

    r = AccuracyResults()
    r.total = 'Overall'

    try:
        check_call(c)
        r.update(parse_conll_eval_table(fp_res))
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
    cc = {}
    ac = {}
    for it in data:
        if it['postag'] not in ac.keys():
            ac[it['postag']] = 0.0
            cc[it['postag']] = 0.0
        if it['postag'] == it['guesstag']:
            cc[it['postag']] += 1
        ac[it['postag']] += 1

    tcc = 0.0
    tac = 0.0

    results = AccuracyResults()

    for t in ac.keys():
        results[t] = (cc[t], ac[t], cc[t] / ac[t])
        tcc += cc[t]
        tac += ac[t]

    results['Total'] = (tcc, tac, tcc / tac)

    return results