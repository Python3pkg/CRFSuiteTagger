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

import time
import eval
import pickle
import readers
import numpy as np

from ftex import FeatureTemplate
from utils import parse_tsv, gsequences, expandpaths
from pycrfsuite import Trainer, Tagger


class CRFSTagger:

    def __init__(self, cfg, ):
        self.cfg = cfg
        expandpaths(self.cfg)
        self.ft_tmpl = None
        self.resources = None
        self.train_data = None
        self.test_data = None

        # loading resources (clusters, embeddings, etc.)
        self.load_resources()

        # loading data
        self.load_data(self.cols)

        # parsing feature template
        self.ft_tmpl = FeatureTemplate()
        self.ft_tmpl.parse_ftvec_templ(self.cfg_tag.get('ftvec'),
                                       self.resources)

    @property
    def cfg_tag(self):
        return dict(self.cfg.items('tagger'))

    @property
    def cfg_crf(self):
        return dict(self.cfg.items('crfsuite'))

    @property
    def cfg_res(self):
        return dict(self.cfg.items('resources'))

    @property
    def ts(self):
        tss = {'\\t': '\t', '\\s': ' '}
        return tss.get(self.cfg_tag['tab_sep'], self.cfg_tag['tab_sep'])

    @property
    def cols(self):
        return self.cfg_tag['cols']

    @property
    def lbl_col(self):
        return self.cfg_tag['label_col']

    @property
    def glbl_col(self):
        return self.cfg_tag['guess_label_col']

    @property
    def model(self):
        return self.cfg_tag['model']

    @property
    def eval_func(self):
        return getattr(eval, '%s' % self.cfg_tag['eval_func'])

    @property
    def verbose(self):
        return bool(self.cfg_tag.get('verbose', True))

    def load_resources(self):
        self.resources = {}
        for n, p in self.cfg_res.items():
            self.resources[n] = getattr(readers, 'read_%s' % n)(p)

    def load_data(self, cols=None):
        c = self.cfg_tag.get('cols', cols)
        if 'train' in self.cfg_tag:
            self.train_data = parse_tsv(self.cfg_tag['train'],
                                        cols=c,
                                        ts=self.ts)

        if 'test' in self.cfg_tag:
            self.test_data = parse_tsv(self.cfg_tag['test'], cols=c, ts=self.ts)

    def extract_features(self, d):

        # number of features
        nft = len(self.ft_tmpl.vec)

        # record count
        rc = len(d)

        # recarray data types (60 >= char string, [30 >= char string] * nft)
        dt = 'a60,{}'.format(','.join('a30' for _ in range(nft)))

        # constructing empty recarray
        fts = np.zeros(rc, dtype=dt)

        # sequence start and end indices
        s, e = 0, 0

        print '%s Processing sequences...' % time.asctime()

        sc = 0

        # extracting features from sequences
        while 0 <= s < len(d):

            # index of the end of a sequence is recorded at the beginning
            e = d[s]['eos']

            # slicing a sequence
            seq = d[s:e]

            # extracting the features
            for i in range(len(seq)):
                fts[s + i] = tuple(self.ft_tmpl.make_fts(seq, i))

            # slicing the feature sequence
            ft_seq = fts[s:e]

            # moving the start index
            s = e

            sc += 1

            if sc % 1000 == 0:
                print '%s processed sequences' % sc

            # yielding a feature sequence
            yield ft_seq

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
        y = gsequences(d, [lc]) if ls is None else ls

        trainer = Trainer(verbose=self.verbose)

        # setting CRFSuite parameters
        trainer.set_params(self.cfg_crf)

        for x_seq, y_seq in zip(X, y):
            trainer.append(x_seq, [l[0] for l in y_seq])

        print '%s Training...' % time.asctime()
        trainer.train(self.model)

        pickle.dump(self.cfg, open('%s.cfg.pcl' % self.model, 'w'))

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

        # checking for a provided tagger
        if tagger is None:
            tgr = Tagger()
            tgr.open(self.model)

        # setting column name for the labels
        lc = self.glbl_col if lbl_col is None else lbl_col

        # tagging
        self.tag(d, tgr, lc)

        #evaluating
        r = self.eval_func(d)

        return r, d

    # a shorthand
    _xfts = extract_features

if __name__ == '__main__':
    print '%s Starting process...' % time.asctime()
    import ConfigParser
    cfg = ConfigParser.ConfigParser()
    cfg.readfp(open('cfg/crfstagger.cfg', 'r'))
    c = CRFSTagger(cfg)
    # trd = parse_data('/Volumes/LocalDataHD/as714/Dropbox/playground/play_corpus_dir/train/tech.pos', ts='\t', cols='pos')
    # tsd = parse_data('/Volumes/LocalDataHD/as714/Dropbox/playground/play_corpus_dir/train/tech.pos', ts='\t', cols='pos')
    # print 'Completed data loading at %s' % time.asctime()
    c.train()
    print '%s Training complete.' % time.asctime()
    print '%s Testing...' % time.asctime()
    r, d = c.test()
    print r
    print time.asctime()