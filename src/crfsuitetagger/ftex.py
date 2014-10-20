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

    def __init__(self, tmpl=None, fnx=None, win_fnx=None):
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
        """

        self.vec = [] if tmpl is None else tmpl
        self.resources = {}

        if win_fnx:
            # all window functions
            wf = win_fnx + [self.emb_win]
            self.win_fnx = {x.__name__: x for x in wf}
        else:
            self.win_fnx = {'emb': FeatureTemplate.emb_win}

        if fnx:
            for f in fnx:
                self.__dict__[f.__name__] = f

    def parse_ftvec_templ(self, s, r):
        """Parses a feature vector template string into a FeatureTemplate object.

        *Important*: if resources (e.g. embeddings) are used in the feature template
        they should be provided during the parsing in the `r` parameter in order to
        be prepacked as parameters to the feature extraction function.

        :param s: feature vector string
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

    def make_fts(self,
                 data,
                 i,
                 form_name='form',
                 *args, **kwargs):
        """

        :param data:
        :param i:
        :param form_name:
        :param args:
        :param kwargs:
        :return:
        """
        ret = [data[i][form_name]]

        # joint attribute dictionary of the class and the instance
        ad = {x: y.__func__ if type(y) is staticmethod else y
              for x, y
              in FeatureTemplate.__dict__.items()}
        ad.update(self.__dict__)

        for itm in self.vec:
            f = itm[0]
            p = itm[1:]
            func = ad[f] if type(f) is str else f
            ret.append(func(data, i, *(p + args), **kwargs))
        return ret

    def add_feature(self, fn, fp=()):
        """Takes a feature extraction function (or its name) and its parameters,
        and packs them into a tuple entry in the feature vector template.

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
        prms = tuple() if fp is None else tuple(fp)
        for i in fw:
            yield (fn, i) + prms

    @staticmethod
    def emb_win(fn, fw, fp, *args, **kwargs):
        """

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
    def word(data, i, rel=0, *args, **kwargs):
        if 0 <= i + rel < len(data):
            form = data[i + rel]['form']
        else:
            form = None
        return 'w[%s]=%s' % (rel, form)

    @staticmethod
    def pos(data, i, rel=0, *args, **kwargs):
        if 0 <= i + rel < len(data):
            postag = data[i + rel]['postag']
        else:
            postag = None
        return 'p[%s]=%s' % (rel, postag)

    @staticmethod
    def chunk(data, i, rel=0, *args, **kwargs):
        if 0 <= i + rel < len(data):
            chunktag = data[i + rel]['chunktag']
        else:
            chunktag = None
        return 'ch[%s]=%s' % (rel, chunktag)

    @staticmethod
    def can(data, i, rel=0, *args, **kwargs):
        if 0 <= i + rel < len(data):
            w = data[i + rel]['form']
            w = re.sub('\d', '#', w)
            w = re.sub('\w', 'x', w)
            w = re.sub('[^#x]', '*', w)
        else:
            w = None
        return 'can[%s]=%s' % (rel, w)

    @staticmethod
    def brown(data, i, rel=0, b=None, p=None, *args, **kwargs):
        """

        :param data: data
        :type data: DataFrame
        :param i: index
        :type i: int
        :param b: brown clusters
        :type b: dict
        :param rel: relative index
        :type rel: int
        :param p: prefix
        :return: feature string
        :rtype str:
        """
        pref = 'full'
        if 0 <= i + rel < len(data):
            try:
                cname = b[data[i + rel]['form']]
                if p:
                    cname = cname[:int(p[0])]
                    pref = p
            except KeyError:
                cname = None
                pref = None
        else:
            cname = None
        return 'cn[%s]:%s=%s' % (rel, pref, cname)

    @staticmethod
    def cls(data, i, rel=0, c=None, *args, **kwargs):
        """

        :param data: data
        :type data: DataFrame
        :param i: index
        :type i: int
        :param c: clusters
        :type c: dict
        :param rel: relative index
        :type rel: int
        :return: feature string
        :rtype: str
        """
        if 0 <= i + rel < len(data):
            try:
                cnum = c[data[i + rel]['form']]
            except KeyError:
                cnum = None
        else:
            cnum = None
        return 'cnum[%s]=%s' % (rel, cnum)

    @staticmethod
    def emb(data, i, rel=0, j=0, e=None, *args, **kwargs):
        if 0 <= i + rel < len(data):
            try:
                emb = e[data[i + rel]['form']][j]
            except KeyError:
                emb = None
        else:
            emb = None
        return 'emb[%s][%s]=%s' % (rel, j, emb)

    @staticmethod
    def isnum(data, i, rel=0, *args, **kwargs):
        if 0 <= i + rel < len(data):
            isnum = bool(re.match('[0-9/]+', data[i + rel]['form']))
        else:
            isnum = None
        return 'isnum[%s]=%s' % (str(rel), isnum)


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