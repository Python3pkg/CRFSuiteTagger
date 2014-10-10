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
import traceback

from os.path import join
from utils import AccuracyResults
from iterpipes import check_call, cmd
from utils import parse_conll_eval_table
from subprocess import CalledProcessError


def conll(data):

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
    data.export_to_file(fp_dp, ['form', 'postag', 'chunktag', 'guesstag'], ' ')

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
        results = parse_conll_eval_table(fp_res)
        r.update(results)
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
    return data.accuracy