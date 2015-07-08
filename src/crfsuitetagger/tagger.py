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
import eval
import pickle
import readers
import shutil
import numpy as np

from os import makedirs
from os.path import dirname, expanduser
from ftex import FeatureTemplate
from utils import parse_tsv, gsequences, expandpaths, clean_cfg
from pycrfsuite import Trainer, Tagger


class CRFSTagger:

    def __init__(self, cfg=None, mp=None, fnx=None, win_fnx=None, cols=None):
        """Creates an instance of CRFSTagger

        :param cfg: configuration
        :type cfg: ConfigParser.ConfigParser
        :param mp: model path
        :type mp: str
        """

        # configuration
        self.cfg = None

        # feature template
        self.ft_tmpl = None

        # list of resources used by features, e.g. word clusters, embeddings
        self.resources = None

        # data
        self.train_data = None
        self.test_data = None

        # instance of pycrfsuite.Tagger
        self.tagger = None

        # attempt to import cannonical replacements
        try:
            import canonical
            self.canonical = canonical.REPLACEMENTS
        except ImportError:
            self.canonical = None

        # load data and resources if configuration is provided
        if cfg:

            self.cfg = cfg
            expandpaths(self.cfg)

            # loading resources (clusters, embeddings, etc.)
            self._load_resources()

            # loading data
            self._load_data()

        # load model
        elif mp:
            m = pickle.load(open(mp, 'r'))
            self.cfg = m.cfg
            self.cfg.set('tagger', 'model', mp)
            self.resources = m.resources

        assert self.cfg is not None, 'Configuration initialisation failed.' \
                                     'Please, provide either a configuration ' \
                                     'or a model.'

        # parsing feature template
        self.ft_tmpl = FeatureTemplate(fnx=fnx, win_fnx=win_fnx, cols=cols)
        self.ft_tmpl.parse_ftvec_templ(self.cfg_tag.get('ftvec'),
                                       self.resources)

    @property
    def cfg_tag(self):
        """Configuration parameters of this tagger. Returns a section from a
        ConfigParser object.


        :return: tagger configuration
        :rtype: dict
        """
        return dict(self.cfg.items('tagger'))

    @property
    def cfg_crf(self):
        """Configuration parameters for CRFSuite. These are passed to the tagger
        instance when training is done. Note, these are not necessarily the
        same as the ones in self.tagger.params.


        :return: CRFSuite configuration
        :rtype: dict
        """
        return dict(self.cfg.items('crfsuite'))

    @property
    def cfg_res(self):
        """Resources configuration. Essentially a list of name and file path
        pairs.


        :return: list of resources
        :rtype: dict
        """
        return dict(self.cfg.items('resources'))

    ############################################################################
    ### A group of properties mapped to configuration values of the tagger.  ###
    ############################################################################
    ############################################################################

    @property
    def ts(self):
        tss = {'\\t': '\t', '\\s': ' '}
        return tss.get(self.cfg_tag['tab_sep'], self.cfg_tag['tab_sep'])

    @property
    def cols(self):
        return self.cfg_tag['cols']

    @property
    def form_col(self):
        return self.cfg_tag.get('form_col', 'form')

    @property
    def lbl_col(self):
        return self.cfg_tag['label_col']

    @property
    def ilbl_col(self):
        return self.cfg_tag.get('guess_label_col', 'guesstag')

    @property
    def model_path(self):
        return self.cfg_tag['model']

    @property
    def eval_func(self):
        return getattr(eval, '%s' % self.cfg_tag['eval_func'])

    @property
    def verbose(self):
        return bool(self.cfg_tag.get('verbose', True))

    @property
    def info(self):
        return self.tagger.info if self.tagger else None

    ############################################################################
    ############################################################################

    def _load_resources(self):
        """Loads resources listed in the `resources` section of the
        configuration. Resources are generally needed for feature generation.
        However, note that for a resource to be loaded a `reader` method
        is needed. For example, to load a clusters resource `cls`, there needs
        to be a method called `read_cls` in `readers.py` that takes a file path
        parameter and returns a resource data structure.
        """
        self.resources = {}
        for n, p in self.cfg_res.items():
            self.resources[n] = getattr(readers, 'read_%s' % n)(p)

    def _load_data(self):
        """Loads training and testing data if provided in the initial
        configuration.
        """
        if 'train' in self.cfg_tag and self.cfg_tag['train']:
            self.train_data = parse_tsv(
                self.cfg_tag['train'],
                cols=self.cols,
                ts=self.ts
            )

        if 'test' in self.cfg_tag and self.cfg_tag['test']:
            self.test_data = parse_tsv(
                fp=self.cfg_tag['test'],
                cols=self.cols,
                ts=self.ts
            )

    def _extract_features(self, doc, form_col='form'):
        """A generator methof that extracts features from the data using a
        feature set template. Yields the feature vector of each sequence in the
        data.

        :param doc: data
        :type doc: np.recarray
        """
        d = copy.deepcopy(doc)

        # replace tokens with canonical forms
        if self.canonical:
            for t in d:
                for r in self.canonical.keys():
                    if re.match(r, t['form']):
                        t['form'] = self.canonical[r]

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

        sc = 0

        # extracting features sequences by sequence
        while 0 <= s < len(d):

            # index of the end of a sequence is recorded at the beginning
            e = d[s]['eos']

            # slicing a sequence
            seq = d[s:e]

            ft_seq = np.zeros(len(seq), dtype=dt)

            # extracting the features
            for i in range(len(seq)):
                ft_seq[i] = tuple(self.ft_tmpl.make_fts(seq, i,
                                                        form_col=form_col))

            # moving the start index
            s = e

            sc += 1

            # yielding a feature sequence
            yield ft_seq

    def train(self, data=None, form_col=None, lbl_col=None, ilbl_col=None,
              data_cols=None, data_sep=None, dump=True):
        """Trains a model based on provided data and features. The default
        behaviour is to load training parameters from the global configuration,
        unless they are passed to this method.

        IMPORTANT: there are two ways to pass data directly through the `data`
        parameter:

        -- np.recarray  `data` needs to be a recarray with column names that
                        match what the feature extractor expects.
        -- csv str      `data` needs to contain a TSV/CSV formatted string.
                        Column names and separator should be provided in the
                        `data_cols` and `data_sep` parameters. They should still
                        match what is expected by the feature extractor.

        The observation, label, and inference column names can be set through
        the global configuration using the following parameter names:
        `form_col`, `label_col`, `guess_label_col`. The default observation
        column name is `fc`, and the inference column name is `guesstag`.
        All three names can be passed to this method to override global
        configuration. Any other column names need to match their respective
        feature extractor functions, e.g. part-of-speech tags need to be placed
        in `postag` column. See `ftex.FeatureTemplate` for others.

        RECOMMENDED: use `utils.parse_tsv` to parse input data to avoid column
        configuration errors.

        NOTE: Due to the way `pycrfsuite` works, the crfsuite model needs to be
        dumped on the hard drive, however, the CRFSuiteTagger model does not
        NEED to be dumped. That process is controlled through the `dump`
        parameter.

        :param data: training data
        :type data: np.recarray or str
        :param form_col: fc column name
        :type form_col: str
        :param lbl_col: label column name
        :type lbl_col: str
        :param ilbl_col: inference label column name
        :type ilbl_col: str
        :param data_cols: list of columns in the data
        :type data_cols: str
        :param data_sep: data tab separator
        :type data_sep: str
        :param dump: dumps the model at specified location if True
        :type dump: bool
        """

        # overriding parameters
        fc = form_col if form_col else self.form_col
        c = data_cols if data_cols else self.cols
        sep = data_sep if data_sep else self.ts
        lc = lbl_col if lbl_col else self.lbl_col
        ilc = ilbl_col if ilbl_col else self.ilbl_col

        if type(data) in [np.core.records.recarray, np.ndarray]:
            d = data
        elif type(data) == str:
            d = parse_tsv(s=data, cols=c, ts=sep, inference_col=ilc)
        elif data is None:
            d = self.train_data
        else:
            raise ValueError('Invalid input type.')

        # extract features
        X = self._extract_features(d, fc)

        # extract labels
        y = gsequences(d, [lc])

        trainer = Trainer(verbose=self.verbose)

        # setting CRFSuite parameters
        trainer.set_params(self.cfg_crf)

        for x_seq, y_seq in zip(X, y):
            trainer.append(x_seq, [l[0] for l in y_seq])

        crfs_mp = '%s.crfs' % self.model_path
        try:
            makedirs(dirname(crfs_mp))
        except OSError:
            pass
        trainer.train(crfs_mp)

        self.tagger = Tagger()
        self.tagger.open(crfs_mp)

        # dumps the model
        if dump:
            self.dump_model(self.model_path)
            pickle.dump(self.cfg, open('%s.cfg.pcl' % self.model_path, 'w'))

    def tag(self, data, form_col=None, ilbl_col=None, tagger=None, cols=None,
            ts=None):
        """Tags TSV/CSV or np.recarray data using the loaded CRFSuite model.

        See documentation for `train` for more details on requirements for the
        data passed to this method.

        :param data: data
        :type data: str or recarray
        :param form_col: form column name
        :type form_col: str
        :param ilbl_col: inference label column name
        :type ilbl_col: str
        :param tagger: CRFS tagger
        :type tagger: Tagger
        :param cols: TSV column names
        :type cols: str or list of str
        :param ts: tab separator for TSV
        :type ts: str
        :return: tagged data
        :rtype: recarray
        """

        fc = form_col if form_col else self.form_col
        c = cols if cols else self.cols
        sep = ts if ts else self.ts
        ilc = ilbl_col if ilbl_col else self.ilbl_col

        if type(data) in [np.core.records.recarray, np.ndarray]:
            d = data
        elif type(data) == str:
            d = parse_tsv(s=data, cols=c, ts=sep)
        else:
            raise ValueError('Invalid input type.')

        tgr = tagger

        if tgr is None and self.tagger:
            tgr = self.tagger
        elif tgr is None:
            tgr = Tagger()
            tgr.open('%s.crfs' % self.model_path)

        # extracting features
        X = self._extract_features(d, form_col=fc)

        # tagging sentences
        idx = 0
        for fts in X:
            for l in tgr.tag(fts):
                d[idx][ilc] = l
                idx += 1

        return d

    def test(self, data=None, form_col=None, ilbl_col=None, tagger=None,
             cols=None, ts=None, eval_func=None):
        """Tags TSV/CSV or np.recarray data using the loaded CRFSuite model and
        evaluates the results.

        See documentation for `train` for more details on requirements for the
        data passed to this method.

        :param data: data
        :type data: str or recarray
        :param form_col: form column name
        :type form_col: str
        :param ilbl_col: inference label column name
        :type ilbl_col: str
        :param tagger: CRFS tagger
        :type tagger: Tagger
        :param cols: TSV column names
        :type cols: str or list of str
        :param ts: tab separator for TSV
        :type ts: str
        :param eval_func: evaluation function name [pos, conll, bio]
        :type eval_func: str
        :return: results and tagged data pair
        :rtype: AccuracyResults, np.recarray
        """

        # use provided data or testing data from config file
        d = self.test_data if data is None else data

        # setting inference label column name
        ilc = self.ilbl_col if ilbl_col is None else ilbl_col

        # tagging
        d = self.tag(d, form_col=form_col, ilbl_col=ilbl_col, tagger=tagger,
                     cols=cols, ts=ts)

        # evaluating
        r = getattr(eval, eval_func) if eval_func else self.eval_func(d)

        # returnning AccuracyResults and np.recarray tagged data
        return r, d

    def dump_model(self, fp):
        """Dumps the CRFSuiteTagger model in provided file path `fp`.

        The dumping consists of two files: <fp> and <fp>.crfs. The first
        contains the configuration and all feature extraction resources needed
        by a CRFSuiteTagger object to replicate this one. The second one is the
        pycrfsuite model that needsto be dumped separately as it is always read
        from the file system.

        :param fp: model file path
        :type fp: str
        """
        md = Model()
        md.cfg = clean_cfg(self.cfg)
        md.resources = self.resources
        fpx = expanduser(fp)
        try:
            makedirs(dirname(fpx))
        except OSError:
            pass
        pickle.dump(md, open(fpx, 'w'))
        if fpx != self.model_path:
            src = '%s.crfs' % self.model_path
            trg = '%s.crfs' % fpx
            try:
                makedirs(dirname(trg))
            except OSError:
                pass
            shutil.copy(src, trg)


class Model:
    """A container class for configuration, feature extraction resources, and
    pycrfsuite model.
    """

    def __init__(self):
        self.crfs_model = None
        self.resources = {}
        self.cfg = None