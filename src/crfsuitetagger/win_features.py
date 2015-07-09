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


def ft_generic_win(fn, fw, fp, *args, **kwargs):
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


def ft_emb_win(fn, fw, fp, *args, **kwargs):
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


def ft_ngram_win(fn, fw, fp, *args, **kwargs):
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