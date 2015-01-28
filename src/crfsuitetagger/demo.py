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

import ConfigParser, logging, os
from tagger import CRFSTagger
from utils import parse_tsv, export

if __name__ == '__main__':

    os.makedirs('tmp')

    # This will train and test a model as configured in a configuration file
    # logging.info('Training and testing a model as configured in
    # cfg/crfstagger.cfg...')
    cfg = ConfigParser.ConfigParser()
    cfg.readfp(open('cfg/crfstagger.cfg', 'r'))
    c = CRFSTagger(cfg)
    c.train()
    r, d = c.test()
    logging.info('Testing complete.')
    print r

    # This model can be dumped in another place
    c.dump_model('tmp/test_model')#

    # This loads a dumped model and uses it to chunk some POS-tagged text
    c = CRFSTagger(mp='tmp/test_model')
    data = parse_tsv('data/sample_chunks.txt', cols='pos', ts='\t')
    d = c.tag(data=data, lc='guesstag', input_type='recarray', cols='chunk')
    export(d, open('tmp/chunk_output.txt', 'w'), cols='chunk')
    print d