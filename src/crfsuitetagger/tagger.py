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

import pickle
import eval
import readers
import ConfigParser

from utils import parse_data, parse_ftvec_templ, SequenceData
from ftex import FeatureContainer
from pycrfsuite import Trainer, Tagger


class CRFSTagger:

    def __init__(self, cp=None, cfg=None):
        if cp:
            cfg_parser = ConfigParser.ConfigParser()
            cfg_parser.readfp(open(cp, 'r'))
        else:
            cfg_parser = cfg
        self.cfg = dict(cfg_parser.items('tagger'))
        self.cfg_crf = dict(cfg_parser.items('crfsuite'))
        self.cfg_res = dict(cfg_parser.items('resources'))
        self.ft_tmpl = None
        self.model = None
        self.resources = None
        self.train_data = None
        self.test_data = None

        self.tagger = None

        tabseps = {'\\t': '\t', '\\s': ' '}
        self.ts = tabseps.get(self.cfg['tab_sep'], self.cfg['tab_sep'])
        self.cols = self.cfg['cols']
        self.lbl_col = self.cfg['label_col']
        self.glbl_col = self.cfg['guess_label_col']
        self.mp = self.cfg['model']
        self.eval_func = getattr(eval, '%s' % self.cfg['eval_func'])

        # loading resources (clusters, embeddings, etc.)
        self.load_resources()

        # loading data
        self.load_data(self.cols)

        # parsing feature template
        self.ft_tmpl = parse_ftvec_templ(self.cfg.get('ftvec'), self.resources)

    def load_resources(self):
        self.resources = {}
        for n, p in self.cfg_res.items():
            self.resources[n] = getattr(readers, 'read_%s' % n)(p)

    def load_data(self, cols=None):
        c = self.cfg.get('cols', cols)
        if self.cfg.has_key('train'):
            self.train_data = parse_data(self.cfg['train'], cols=c, ts=self.ts)

        if self.cfg.has_key('test'):
            self.test_data = parse_data(self.cfg['test'], cols=c, ts=self.ts)

    def extract_features(self, d):

        fc = FeatureContainer(d)
        fc.extract_features(self.ft_tmpl)

        return fc.gsequences

    @staticmethod
    def get_labels(d, lc):
        lbls = []
        for s in d:
            lbls.append([getattr(x, lc) for x in s])

    def dump_ft_template(self, fp):
        pickle.dump(self.ft_tmpl, fp)

    def dump_fts(self, fp, data=None):
        d = self.train_data if data is None else data
        ft = list(self._xfts(d))
        pickle.dump(ft, fp)

    def train(self, data=None, fts=None, ls=None, lbl_col=None):

        # setting up the training data
        d = self.train_data if data is None else data

        # setting label column name
        lc = self.lbl_col if lbl_col is None else lbl_col

        # extract features or use provided
        import time
        print '%s Extracting features...' % time.asctime()
        X = self._xfts(d) if fts is None else fts

        # extract labels or use provided
        print '%s extracting labels' % time.asctime()
        y = SequenceData(d).gsequences(lc) if ls is None else ls

        trainer = Trainer(verbose=False)

        # setting CRFSuite parameters
        trainer.set_params(self.cfg_crf)

        for x_seq, y_seq in zip(X, y):
            trainer.append(x_seq, y_seq)

        print '%s Training...' % time.asctime()
        trainer.train(self.mp)

    def tag(self, data, tagger, lc):

        # extracting features
        X = self.extract_features(data)

        # tagging sentences
        idx = 0
        for fts in X:
            for l in tagger.tag(fts):
                data[idx][lc] = l
                idx += 1

        return data

    def test(self, data=None, tagger=None, lbl_col=None):

        # use provided data or testing data from config file
        d = self.test_data if data is None else data

        # checking if there is a trained or provided tagger
        if self.tagger is None and tagger is None:
            self.tagger = Tagger()
            self.tagger.open(self.mp)

        # assigning tagger
        tgr = self.tagger if tagger is None else tagger

        # setting column name for the labels
        lc = self.glbl_col if lbl_col is None else lbl_col

        # tagging
        self.tag(d, tgr, lc)

        #evaluating
        r = self.eval_func(SequenceData(d))

        return r, d

    # a shorthand
    _xfts = extract_features

if __name__ == '__main__':
    import time
    print '%s Starting process...' % time.asctime()
    c = CRFSTagger('cfg/crfstagger.cfg')
    # trd = parse_data('/Volumes/LocalDataHD/as714/Dropbox/playground/play_corpus_dir/train/tech.pos', ts='\t', cols='pos')
    # tsd = parse_data('/Volumes/LocalDataHD/as714/Dropbox/playground/play_corpus_dir/train/tech.pos', ts='\t', cols='pos')
    # print 'Completed data loading at %s' % time.asctime()
    c.train()
    print '%s Training complete.' % time.asctime()
    print '%s Testing...' % time.asctime()
    r, d = c.test()
    print d[:100]
    print r
    print time.asctime()