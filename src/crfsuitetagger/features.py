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


def ft_word(data, i, cols, rel=0, *args, **kwargs):
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


def ft_nword(data, i, cols, rel=0, n=2, *args, **kwargs):
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
    if 0 <= i + rel and i + rel + int(n) - 1 < len(data):
        s = i + rel
        e = i + rel + int(n)
        forms = ''.join([data[x][cols['form']] for x in range(s, e)])
    else:
        forms = None
    return '%sw[%s]=%s' % (n, rel, forms)


def ft_pos(data, i, cols, rel=0, *args, **kwargs):
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


def ft_npos(data, i, cols, rel=0, n=2, *args, **kwargs):
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
    if 0 <= i + rel and i + rel + int(n) - 1 < len(data):
        s = i + rel
        e = i + rel + int(n)
        postags = ''.join([data[x][cols['postag']] for x in range(s, e)])
    else:
        postags = None
    return '%sp[%s]=%s' % (n, rel, postags)


def ft_chunk(data, i, cols, rel=0, *args, **kwargs):
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


def ft_nchunk(data, i, cols, rel=0, n=None, *args, **kwargs):
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


def ft_can(data, i, cols, rel=0, *args, **kwargs):
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


def ft_brown(data, i, cols, rel=0, b=None, p=None, *args, **kwargs):
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


def ft_cls(data, i, cols, rel=0, c=None, *args, **kwargs):
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


def ft_emb(data, i, cols, rel=0, j=0, e=None, *args, **kwargs):
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


def ft_lex(data, i, cols, rel=0, lex=None, *args, **kwargs):
    """Generates binary features based on the presence of a word form in a
    dictionary resource (lexicon).

    :param data: data
    :type data: DataFrame
    :param i: index
    :type i: int
    :param cols: column map
    :type cols: dict
    :param rel: relative index
    :type rel: int
    :param lex: lexicon
    :type lex: set or list
    :return: feature string
    :rtype: str
    """
    lex_ft = False
    if 0 <= i + rel < len(data):
        try:
            lex_ft = data[i + rel][cols['form']] in lex
        except KeyError:
            pass

    return 'lex[%s]=%s' % (rel, lex_ft)


def ft_nem(data, i, cols, rel=0, *args, **kwargs):

    nem_ft = None
    if 0 <= i + rel < len(data):
        try:
            nem_ft = data[i + rel][cols['nemeta']]
        except KeyError:
            pass
    return 'nem[%s]=%s' % (rel, nem_ft)


def ft_nsuff(data, i, cols, rel=0, n=3, *args, **kwargs):
    nsuff_ft = None
    if 0 <= i + rel < len(data):
        try:
            nsuff_ft = data[i + rel][cols['form']][-int(n):]
        except KeyError:
            pass
    return 'nsuff[%s],%s=%s' % (rel, n, nsuff_ft)


def ft_npref(data, i, cols, rel=0, n=3, *args, **kwargs):
    npref_ft = None
    if 0 <= i + rel < len(data):
        try:
            npref_ft = data[i + rel][cols['form']][:-int(n)]
        except KeyError:
            pass
    return 'npref[%s],%s=%s' % (rel, n, npref_ft)



def ft_isnum(data, i, cols, rel=0, *args, **kwargs):
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


def ft_short(data, i, cols, rel=0, p=2, *args, **kwargs):
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


def ft_long(data, i, cols, rel=0, p=12, *args, **kwargs):
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


def ft_ln(data, i, cols, rel=0, *args, **kwargs):
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


def ft_suff(data, i, cols, rel=0, sfxs=None, max_sfx=0, *args, **kwargs):
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


def ft_pref(data, i, cols, rel=0, prfxs=None, max_prfx=0, *args, **kwargs):
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


def ft_medpref(data, i, cols, rel=0, prfxs=None, max_prfx=0, *args, **kwargs):
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
    return 'med%s' % ft_pref(data, i, cols, rel, prfxs, max_prfx)


def ft_medsuff(data, i, cols, rel=0, sfxs=None, max_sfx=0, *args, **kwargs):
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
    return 'med%s' % ft_suff(data, i, cols, rel, sfxs, max_sfx)


def ft_nounsuff(data, i, cols, rel=0, sfxs=None, max_sfx=0, *args, **kwargs):
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
    return 'noun%s' % ft_suff(data, i, cols, rel, sfxs, max_sfx)


def ft_verbsuff(data, i, cols, rel=0, sfxs=None, max_sfx=0, *args, **kwargs):
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
    return 'verb%s' % ft_suff(data, i, cols, rel, sfxs, max_sfx)


def ft_adjsuff(data, i, cols, rel=0, sfxs=None, max_sfx=0, *args, **kwargs):
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
    return 'adj%s' % ft_suff(data, i, cols, rel, sfxs, max_sfx)


def ft_advsuff(data, i, cols, rel=0, sfxs=None, max_sfx=0, *args, **kwargs):
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
    return 'adv%s' % ft_suff(data, i, cols, rel, sfxs, max_sfx)


def ft_inflsuff(data, i, cols, rel=0, sfxs=None, max_sfx=0, *args, **kwargs):
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
    return 'infl%s' % ft_suff(data, i, cols, rel, sfxs, max_sfx)