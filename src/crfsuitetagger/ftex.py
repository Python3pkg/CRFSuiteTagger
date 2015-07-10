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
import features as fts
import win_features as wf


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

        # grouping feature functions
        self.fnx = {
            x[3:]: y for x, y in fts.__dict__.items() if x[:3] == 'ft_'
        }
        if fnx:
            self.fnx.update({f.__name__: f for f in fnx})

        # grouping window feature functions
        self.win_fnx = {
            x[3:-4]: y for x, y in wf.__dict__.items() if x[:3] == 'ft_'
        }
        if win_fnx:
            self.win_fnx.update({x.__name__: x for x in win_fnx})

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
            fw = wf.parse_range(m.group(1)) if m.group(1) else None

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
        f = wfnx[wfn] if wfn in wfnx.keys() else wf.ft_generic_win
        for v in f(fn, fw, fp, *args, **kwargs):
            self.vec.append(v)

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

        for itm in self.vec:
            f = itm[0]
            p = itm[1:]
            func = self.fnx[f] if type(f) is str else f
            ret.append(func(data, i, self.cols, *(p + args), **kwargs))
        return ret