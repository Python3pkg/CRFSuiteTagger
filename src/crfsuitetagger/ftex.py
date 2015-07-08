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


class FeatureTemplate:

    def __init__(self, tmpl=None, fnx=None, win_fnx=None, cols=None):
        """Constructs either a FeatureTemplate object or takes parameters to
        set the template dictionary and the list of special functions.

        *Template dictionary:*

        The template dictionary should consist of an arbitrary key and a list of
        values starting with the name of a feature function.

        *Feature extraction functions:*

        Additional feature extraction functions can be provided during the
        construction of an object through the `fnx` parameter.

        Most feature functions generate context-based features that need input
        from one or more of the data fields of a data point (single features) or
        a context window (window features). In practice, all window features
        generate a number or single features iterating over the indices in their
        window. Most window feature functions generate one feature per context
        window data point, e.g. word[-2]=foo or postag[-1]=NN. Such behaviour is
        handled by the `generic_win` method. In case more than one feature
        should be generated, a special window function has to be provided as
        well through the `win_fnx` parameter.

        :param tmpl: template
        :type tmpl: list
        :param fnx: additional feature extraction functions
        :type fnx: list
        :param win_fnx: additional window feature extraction functions
        :type win_fnx: list
        :param cols: map of columns names
        :type cols: dict
        """

        self.vec = [] if tmpl is None else tmpl
        self.resources = {}
        self.cols = cols if cols else {'form': 'form', 'postag': 'postag',
                                       'chunktag': 'chunktag', 'netag': 'netag'}

        if win_fnx:
            # all window functions
            wf = win_fnx + [self.emb_win]
            self.win_fnx = {x.__name__: x for x in wf}
        else:
            self.win_fnx = {
                'emb': FeatureTemplate.emb_win,
                'nword': FeatureTemplate.ngram_win,
                'npos': FeatureTemplate.ngram_win,
                'nchunk': FeatureTemplate.ngram_win
            }

        if fnx:
            for f in fnx:
                self.__dict__[f.__name__] = f

    def parse_ftvec_templ(self, s, r):
        """Parses a feature vector template string into a FeatureTemplate
        object.

        *Important*: if resources (e.g. embeddings) are used in the feature
        template they should be provided during the parsing in the `r`
        parameter in order to be prepacked as parameters to the feature
        extraction function.

        Feature vector template example:

        word:[-1:1];pos:[-1:0];npos:[-1:1],2;cls:[0];short

        Note that `cls` requires a resource (see `cls` function) and `npos` has
        an additional function parameter indicating bigram features should be
        generated (as opposed to other n-grams). Additional attributes may have
        default values used in case they are omitted in the feature template.

        **FEATURE VECTOR TEMPLATE BUILDING FUNCTION**

        :param s: feature vector template string
        :type s: str
        :param r: dictionary of resources
        :type r: dict
        :return: FeatureTemplate
        """
        fts_str = [x for x in re.sub('[\t ]', '', s).split(';')]
        for ft in fts_str:

            # empty featues (...; ;feature:params)
            if ft.strip() == '':
                continue

            # no parameter features
            no_par = ':' not in ft
            # misplaced column without parameters
            no_par_end_col = ft.count(':') == 1 and ft.endswith(':')
            if no_par or no_par_end_col:
                fn = ft if no_par else ft[:-1]
                self.add_feature(fn)
                continue

            # function name & parameter values
            fn, v = ft.split(':', 1)

            # value matches
            m = re.match('(?:\[([0-9:,-]+)\])?(.+)?', v)

            # window range
            fw = parse_range(m.group(1)) if m.group(1) else None

            # function parameters
            fp = []

            # adding resources to the parameters if required
            if fn in r.keys():
                fp.append(r[fn])

            # adding function parameters if specified
            if m.group(2) is not None:
                fp.extend([x for x in m.group(2).split(',') if x])

            # name, window, parameters
            self.add_win_features(fn, fw, tuple(fp))

    def add_feature(self, fn, fp=()):
        """Takes a feature extraction function (or its name) and its parameters,
        and packs them into a tuple entry in the feature vector template.

        **FEATURE VECTOR TEMPLATE BUILDING FUNCTION**

        :param fn: feature function name
        :type fn: str or function
        :param fp: feature function parameters
        :type fp: tuple
        """
        self.vec.append((fn,) + fp)

    def add_win_features(self, fn, fw, fp, *args, **kwargs):
        """Takes a feature extraction function (or its name), a generator of
        context window indices, and a tuple containing the function parameters.
        It iterates over the generator adding entries to the feature vector
        template.

        **FEATURE VECTOR TEMPLATE BUILDING FUNCTION**

        :param fn: feature extraction function name
        :type fn: function or str
        :param fw: feature extraction function window generator of indices
        :type fw: generator
        :param fp: feature extraction function parameters
        :type fp: tuple
        """
        wfn = fn if type(fn) is str else fn.__name__
        wfnx = self.win_fnx
        f = wfnx[wfn] if wfn in wfnx.keys() else self.generic_win
        for v in f(fn, fw, fp, *args, **kwargs):
            self.vec.append(v)

    @staticmethod
    def generic_win(fn, fw, fp, *args, **kwargs):
        """Iterates over the list of single features that make up a context
        window feature, and yields them one at a time. This function is the
        default behaviour for context window features.

        Note: If a window feature requires special behaviour, another window
        function needs to be provided and linked to it in the constructor. See
        `fnx` and `win_fnx` attributes in the constructor.

        **FEATURE VECTOR TEMPLATE BUILDING FUNCTION**

        :param fn: function name
        :param fw: context window
        :param fp: additional feature function parameters
        """
        prms = tuple() if fp is None else tuple(fp)
        for i in fw:
            yield (fn, i) + prms

    @staticmethod
    def emb_win(fn, fw, fp, *args, **kwargs):
        """Same as `generic_win`, but suited for embeddings features.

        **FEATURE VECTOR TEMPLATE BUILDING FUNCTION**

        :param fn: function name
        :param fw: embeddings window (range of ints)
        :param fp: embeddings
        """

        # embeddings
        e = fp[0]

        # vector coverage
        if len(fp) > 1:
            # parse specified range of the embeddings vector
            vc = parse_range(fp[1][1:-1])
        else:
            # assume iteration over the whole vector
            vc = range(len(e[e.keys()[0]]))

        for i in fw:
            for j in vc:
                yield (fn, i, j, e)

    @staticmethod
    def ngram_win(fn, fw, fp, *args, **kwargs):
        """Yields the starting indices of all full n-grams from left to right.

        **FEATURE VECTOR TEMPLATE BUILDING FUNCTION**

        :param fn: function name
        :param fw: n-grams window
        :param fp: feature params (position 0 reserved for n in n-grams)
        """
        try:
            n = int(fp[0])
        except IndexError:
            # in case no parameter is provided, bigrams are used
            n = 2
        prms = tuple() if len(fp) == 1 else tuple(fp[1:])
        nfw = parse_ng_range(fw, n)
        for i in nfw:
            yield (fn, i, n) + prms

    def make_fts(self,
                 data,
                 i,
                 form_col='form',
                 *args, **kwargs):
        """Generates the (context) features for a single item in a sequence
        based on the feature template embedded in this object.

        **FEATURE VECTOR GENERATING FUNCTION**

        :param data: data sequence
        :type data: np.recarray
        :param i: index
        :type i: int
        :param form_col: name of column containing the form
        :type form: str
        :return: feature matrix
        :rtype: list
        """
        ret = [data[i][form_col]]

        # joint attribute dictionary of the class and the instance
        # needed for joint access to implemented and user-provided methods
        ad = {x: y.__func__ if type(y) is staticmethod else y
              for x, y
              in FeatureTemplate.__dict__.items()}
        ad.update(self.__dict__)

        for itm in self.vec:
            f = itm[0]
            p = itm[1:]
            func = ad[f] if type(f) is str else f
            ret.append(func(data, i, self.cols, *(p + args), **kwargs))
        return ret

    @staticmethod
    def word(data, i, cols, rel=0, *args, **kwargs):
        """Generates a feature based on the `form` column.

        **FEATURE GENERATION FUNCTION**

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :return: feature
        :rtype: str
        """
        if 0 <= i + rel < len(data):
            form = data[i + rel][cols['form']]
        else:
            form = None
        return 'w[%s]=%s' % (rel, form)

    @staticmethod
    def nword(data, i, cols, rel=0, n=None):
        """Generates a n-gram context feature based on the `form` column.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :param n: n in n-gram
        :type n: int
        :return: feature
        :rtype: str
        """
        if 0 <= i + rel and i + rel + n - 1 < len(data):
            s = i + rel
            e = i + rel + n
            forms = ''.join([data[x][cols['form']] for x in range(s, e)])
        else:
            forms = None
        return '%sw[%s]=%s' % (n, rel, forms)

    @staticmethod
    def pos(data, i, cols, rel=0, *args, **kwargs):
        """Generates a context feature based on part of speech in column `pos`.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :return: feature
        :rtype: str
        """
        if 0 <= i + rel < len(data):
            postag = data[i + rel][cols['postag']]
        else:
            postag = None
        return 'p[%s]=%s' % (rel, postag)

    @staticmethod
    def npos(data, i, cols, rel=0, n=2):
        """Generates a n-gram context feature based on the `postag` column.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :param n: n in n-gram
        :type n: int
        :return: feature
        :rtype: str
        """
        if 0 <= i + rel and i + rel + n - 1 < len(data):
            s = i + rel
            e = i + rel + n
            postags = ''.join([data[x]['postag'] for x in range(s, e)])
        else:
            postags = None
        return '%sp[%s]=%s' % (n, rel, postags)

    @staticmethod
    def chunk(data, i, cols, rel=0, *args, **kwargs):
        """Generates a context feature based on chunk annotation in column
        `chunktag`.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :return: feature
        :rtype: str
        """
        if 0 <= i + rel < len(data):
            chunktag = data[i + rel][cols['chunktag']]
        else:
            chunktag = None
        return 'ch[%s]=%s' % (rel, chunktag)

    @staticmethod
    def nchunk(data, i, cols, rel=0, n=None):
        """Generates a n-gram context feature based on the `chunktag` column.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :param n: n in n-gram
        :type n: int
        :return: feature
        :rtype: str
        """
        if 0 <= i + rel and i + rel + n - 1 < len(data):
            s = i + rel
            e = i + rel + n
            chunktags = ''.join(
                [data[x][cols['chunktag']] for x in range(s, e)]
            )
        else:
            chunktags = None
        return '%sp[%s]=%s' % (n, rel, chunktags)

    @staticmethod
    def can(data, i, cols, rel=0, *args, **kwargs):
        """Generates a context feature based on canonicalised form of the
        `form` column.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :return: feature
        :rtype: str
        """
        if 0 <= i + rel < len(data):
            w = data[i + rel][cols['form']]
            w = re.sub('\d', '#', w)
            w = re.sub('\w', 'x', w)
            w = re.sub('[^#x]', '*', w)
        else:
            w = None
        return 'can[%s]=%s' % (rel, w)

    @staticmethod
    def brown(data, i, cols, rel=0, b=None, p=None, *args, **kwargs):
        """Generates Brown (hierarchical) clusters feature based on the `form`
        column value.

        See link for more details on data resource format:

            https://github.com/percyliang/brown-cluster

        :param data: data
        :type data: DataFrame
        :param i: index
        :type i: int
        :param cols: column map
        :type cols: dict
        :param b: brown clusters
        :type b: dict
        :param rel: relative index
        :type rel: int
        :param p: prefix
        :return: feature string
        :rtype str:
        """
        cname = None
        if 0 <= i + rel < len(data):
            try:
                cname = b[data[i + rel][cols['form']]]
                if p:
                    cname = cname[:int(p)]
            except KeyError:
                pass
        pref = p if p else 'full'
        return 'cn[%s]:%s=%s' % (rel, pref, cname)

    @staticmethod
    def cls(data, i, cols, rel=0, c=None, *args, **kwargs):
        """Generates features from flat word clusters based on the `form`
        column.

        See link for more details on data resource format:

            https://github.com/ninjin/clark_pos_induction

        :param data: data
        :type data: DataFrame
        :param i: index
        :type i: int
        :param cols: column map
        :type cols: dict
        :param c: clusters
        :type c: dict
        :param rel: relative index
        :type rel: int
        :return: feature string
        :rtype: str
        """
        if 0 <= i + rel < len(data):
            try:
                cnum = c[data[i + rel][cols['form']]]
            except KeyError:
                cnum = None
        else:
            cnum = None
        return 'cnum[%s]=%s' % (rel, cnum)

    @staticmethod
    def emb(data, i, cols, rel=0, j=0, e=None, *args, **kwargs):
        """Generates features from word embeddings based on the `form` column.

        See links for more details on data resource format:

            http://metaoptimize.com/projects/wordreprs/
            https://code.google.com/p/word2vec/

        GOTCHA: some resources come with separators of 4 space characters
        (replacing a tab?), while the default is a single space.

        :param data: data
        :type data: DataFrame
        :param i: index
        :type i: int
        :param cols: column map
        :type cols: dict
        :param c: clusters
        :type c: dict
        :param rel: relative index
        :type rel: int
        :return: feature string
        :rtype: str
        """
        if 0 <= i + rel < len(data):
            try:
                emb = e[data[i + rel][cols['form']]][j]
            except KeyError:
                emb = None
        else:
            emb = None
        return 'emb[%s][%s]=%s' % (rel, j, emb)

    @staticmethod
    def isnum(data, i, cols, rel=0, *args, **kwargs):
        """Generates a boolean context feature based on weather the value of
        the `form` column is a number.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :return: feature
        :rtype: str
        """
        if 0 <= i + rel < len(data):
            isnum = bool(re.match('[0-9/]+', data[i + rel][cols['form']]))
        else:
            isnum = None
        return 'isnum[%s]=%s' % (str(rel), isnum)

    @staticmethod
    def short(data, i, cols, rel=0, p=2, *args, **kwargs):
        """Generates a context feature based on the length of the value the
        `form` column. Positive if value is shorted than the provided threshold.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :param p: threshold
        :type p: int
        :return: feature
        :rtype: str
        """
        shrt = None
        if 0 <= i + rel < len(data):
            shrt = len(data[i + rel][cols['form']]) < p
        return 'short[%s]=%s' % (str(rel), shrt)

    @staticmethod
    def long(data, i, cols, rel=0, p=12, *args, **kwargs):
        """Generates a context feature based on the length of the value the
        `form` column. Positive if value is longer than provided threshold.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :param p: threshold
        :type p: int
        :return: feature
        :rtype: str
        """
        lng = None
        if 0 <= i + rel < len(data):
            lng = len(data[i + rel][cols['form']]) > p
        return 'long[%s]=%s' % (str(rel), lng)

    @staticmethod
    def ln(data, i, cols, rel=0, *args, **kwargs):
        """Generates a context feature based on the length of the value the
        `form` column.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :return: feature
        :rtype: str
        """
        ln = None
        if 0 <= i + rel < len(data):
            ln = len(data[i + rel][cols['form']])
        return 'ln[%s]=%s' % (str(rel), ln)

    @staticmethod
    def suff(data, i, cols, rel=0, sfxs=None, max_sfx=0, *args, **kwargs):
        """Generates a context feature based on the longest possible suffix of
        the value in the `form` column. The suffix is only valid if present in
        a list of suffixes.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :param sfxs: suffixes
        :type sfxs: set
        :param max_sfx: max suffix length
        :type max_sfx: int
        :return: feature
        :rtype: str
        """
        sufx = None
        if 0 <= i + rel < len(data):
            w = data[i + rel][cols['form']]
            maxs = len(w) - 1
            if max_sfx and int(max_sfx) < maxs:
                maxs = int(max_sfx)
            # TODO turn this loop around for efficiency.
            for s in (w[-x:] for x in range(1, maxs + 1)):
                if s in sfxs:
                    sufx = s  # longest possible suffix
        return 'sfx[%s]=%s' % (str(rel), sufx)

    @staticmethod
    def pref(data, i, cols, rel=0, prfxs=None, max_prfx=0, *args, **kwargs):
        """Generates a context feature based on the longest possible prefix of
        the value in the `form` column. The prefix is only valid if present in
        a list of prefixes.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :param prfxs: prefixes
        :type prfxs: set
        :param max_prfx: max prefix length
        :type max_prfx: int
        :return: feature
        :rtype: str
        """
        prfx = None
        if 0 <= i + rel < len(data):
            w = data[i + rel][cols['form']]
            maxp = len(w)
            if max_prfx and int(max_prfx) < maxp:
                maxp = int(max_prfx)
            # TODO turn this loop around for efficiency.
            for s in (w[:x] for x in range(1, maxp + 1)):
                if s in prfxs:
                    prfx = s  # longest possible suffix
        return 'sfx[%s]=%s' % (str(rel), prfx)

    @staticmethod
    def medpref(data, i, cols, rel=0, prfxs=None, max_prfx=0, *args, **kwargs):
        """Generates a context feature based on the longest possible prefix of
        the value in the `form` column. The prefix is only valid if present in
        a list of medical prefixes.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :param prfxs: prefixes
        :type prfxs: set
        :param max_prfx: max prefix length
        :type max_prfx: int
        :return: feature
        :rtype: str
        """
        return 'med%s' % FeatureTemplate.pref(data, i, cols, rel, prfxs, max_prfx)

    @staticmethod
    def medsuff(data, i, cols, rel=0, sfxs=None, max_sfx=0, *args, **kwargs):
        """Generates a context feature based on the longest possible suffix of
        the value in the `form` column. The suffix is only valid if present in
        a list of suffixes.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :param sfxs: suffixes
        :type sfxs: set
        :param max_sfx: max suffix length
        :type max_sfx: int
        :return: feature
        :rtype: str
        """
        return 'med%s' % FeatureTemplate.suff(data, i, cols, rel, sfxs, max_sfx)

    @staticmethod
    def nounsuff(data, i, cols, rel=0, sfxs=None, max_sfx=0, *args, **kwargs):
        """Generates a context feature based on the longest possible suffix of
        the value in the `form` column. The suffix is only valid if present in
        a list of suffixes.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :param sfxs: suffixes
        :type sfxs: set
        :param max_sfx: max suffix length
        :type max_sfx: int
        :return: feature
        :rtype: str
        """
        return 'noun%s' % FeatureTemplate.suff(data, i, cols, rel, sfxs, max_sfx)

    @staticmethod
    def verbsuff(data, i, cols, rel=0, sfxs=None, max_sfx=0, *args, **kwargs):
        """Generates a context feature based on the longest possible suffix of
        the value in the `form` column. The suffix is only valid if present in
        a list of suffixes.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :param sfxs: suffixes
        :type sfxs: set
        :param max_sfx: max suffix length
        :type max_sfx: int
        :return: feature
        :rtype: str
        """
        return 'verb%s' % FeatureTemplate.suff(data, i, cols, rel, sfxs, max_sfx)

    @staticmethod
    def adjsuff(data, i, cols, rel=0, sfxs=None, max_sfx=0, *args, **kwargs):
        """Generates a context feature based on the longest possible suffix of
        the value in the `form` column. The suffix is only valid if present in
        a list of suffixes.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :param sfxs: suffixes
        :type sfxs: set
        :param max_sfx: max suffix length
        :type max_sfx: int
        :return: feature
        :rtype: str
        """
        return 'adj%s' % FeatureTemplate.suff(data, i, cols, rel, sfxs, max_sfx)

    @staticmethod
    def advsuff(data, i, cols, rel=0, sfxs=None, max_sfx=0, *args, **kwargs):
        """Generates a context feature based on the longest possible suffix of
        the value in the `form` column. The suffix is only valid if present in
        a list of suffixes.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :param sfxs: suffixes
        :type sfxs: set
        :param max_sfx: max suffix length
        :type max_sfx: int
        :return: feature
        :rtype: str
        """
        return 'adv%s' % FeatureTemplate.suff(data, i, cols, rel, sfxs, max_sfx)

    @staticmethod
    def inflsuff(data, i, cols, rel=0, sfxs=None, max_sfx=0, *args, **kwargs):
        """Generates a context feature based on the longest possible suffix of
        the value in the `form` column. The suffix is only valid if present in
        a list of suffixes.

        :param data: data
        :type: np.recarray
        :param i: focus position
        :type i: int
        :param cols: column map
        :type cols: dict
        :param rel: relative position of context features
        :type rel: int
        :param sfxs: suffixes
        :type sfxs: set
        :param max_sfx: max suffix length
        :type max_sfx: int
        :return: feature
        :rtype: str
        """
        return 'infl%s' % FeatureTemplate.suff(data, i, cols, rel, sfxs, max_sfx)


def parse_range(r):
    """Parses a range in string representation adhering to the following
    format:
    1:3,6,8:9 -> 1,2,3,6,8,9

    :param r: range string
    :type r: str
    """
    rng = []

    # Range strings
    rss = [x.strip() for x in r.split(',')]

    for rs in rss:
        if ':' in rs:
            # Range start and end
            s, e = (int(x.strip()) for x in rs.split(':'))
            for i in range(s, e + 1):
                rng.append(int(i))
        else:
            rng.append(int(rs))

    return rng


def nrange(start, stop, step):
    """Returns the indices of n-grams in a context window. Works much like
    range(start, stop, step), but the stop index is inclusive, and indices are
    included only if the step can fit between the candidate index and the stop
    index.

    :param start: starting index
    :type start: int
    :param stop: stop index
    :type stop: int
    :param step: n-gram length
    :type step: int
    :return: n-gram indices from left to right
    :rtype: list of int
    """
    idx = start
    rng = []
    while idx + step <= stop + 1:
        rng.append(idx)
        idx += 1
    return rng


def parse_ng_range(fw, n):
    """Transforms context window index list to a context window n-gram index
    list.

    :param fw: context window
    :type fw: list of int
    :param n: n in n-grams
    :type n: int
    :return: n-gram indices
    :rtype: list of int
    """
    subranges = []
    cur = None
    rng = []
    for i in fw:
        if cur == None or cur + 1 == i:
            rng.append(i)
            cur = i
        else:
            subranges.append(rng)
            rng = [i]
            cur = i
    subranges.append(rng)
    nrng = []
    for sr in subranges:
        for i in nrange(sr[0], sr[-1], n):
            nrng.append(i)
    return nrng
