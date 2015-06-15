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
import time
import StringIO
import copy
from unittest import TestCase

from crfsuitetagger.ftex import *
from crfsuitetagger.utils import *
from crfsuitetagger.eval import *


class TestUtils(TestCase):
    def setUp(self):
        self.data_str = '''The\tD
quick\tA
fox\tN
jumped\tV
across\tR
the\tD
river\tN
.\t.

The\tD
stupid\tA
wolf\tN
fell\tV
in\tI
the\tD
trap\tN
.\t.'''
        self.dp = 'tmp/data.%s.tmp' % time.asctime()
        open(self.dp, 'w').write(self.data_str)
        self.dp_large = 'tmp/data_large.%s.tmp' % time.asctime()
        with open(self.dp_large, 'w') as fh:
            fh.write('\n\n'. join([self.data_str + '\n.\t.' for _ in range(10)]))
        self.data = parse_tsv(self.dp, ('form', 'postag'))
        self.data_large = parse_tsv(self.dp_large, ('form', 'postag'))

    def tearDown(self):
        os.remove(self.dp)

    def test_parse_tsv(self):
        dt = 'a60,a10,a10,int32'
        mock_data = np.zeros(16, dtype=dt)
        mock_data.dtype.names = ('form', 'postag', 'guesstag', 'eos')
        for idx, line in enumerate(
                x for x in self.data_str.split('\n') if x.strip()):
            if '\t' in line:
                mock_data[idx] = tuple(line.strip().split('\t')) + ('', -1)
        mock_data[0]['eos'] = 8
        mock_data[8]['eos'] = 16

        self.assertItemsEqual(self.data, mock_data)

    def test_count_records(self):
        # in case the string is changed
        rc = len([x for x in self.data_str.strip().split('\n') if x.strip()])
        self.assertEqual(count_records(open(self.dp, 'r')), rc)
        fp = 'tmp/count.%s.tmp' % time.asctime()
        open(fp, 'w').write('%s\n\n' % self.data_str)
        self.assertEqual(count_records(open(fp, 'r')), rc)
        os.remove(fp)

    def test_export(self):
        ex = StringIO.StringIO()
        export(self.data, ex, cols=['form', 'postag'])
        self.assertEqual(ex.getvalue().strip(), self.data_str)

    def test_gsequence(self):
        gs = [list(x) for x in gsequences(self.data)]
        dt = 'a60,a10,a10,int32'
        mock_data = np.zeros(16, dtype=dt)
        mock_data.dtype.names = ('form', 'postag', 'guesstag', 'eos')
        for idx, line in enumerate(
                x for x in self.data_str.split('\n') if x.strip()):
            if '\t' in line:
                mock_data[idx] = tuple(line.strip().split('\t')) + ('', -1)
        mock_data[0]['eos'] = 8
        mock_data[8]['eos'] = 16

        mgs = [list(mock_data[:8]), list(mock_data[8:])]

        self.assertSequenceEqual(gs, mgs)

    def test_count(self):
        self.assertEqual(count_sequences(self.data), 2)
        self.assertEqual(count_sequences(self.data_large), 20)

    def test_weighted_split(self):
        trd, ted = weighed_split(self.data_large, proportion=0.89)
        self.assertEqual(len(trd), 144)
        self.assertEqual(len(ted), 26)
        self.assertEqual(ted[0]['eos'], 9)
        self.assertEqual(trd[0]['eos'], 8)
        s = 0
        sh = 0
        while s < len(trd):
            s = trd[s]['eos']
            self.assertNotEqual(s, sh)
            sh = s
        self.assertEqual(s, len(trd))
        s = 0
        sh = 0
        while s < len(ted):
            s = ted[s]['eos']
            self.assertNotEqual(s, sh)
            sh = s
        self.assertEqual(s, len(ted))

    def test_set_seq_start_idx(self):
        d = copy.deepcopy(self.data_large)
        eos_idx = 1
        while d[eos_idx]['eos'] < 0:
            eos_idx += 1
        idx = 10
        set_sequence_start_idx(d, idx)
        s = 0
        sh = 0
        while s < len(d):
            s = d[s]['eos'] - idx
            self.assertNotEqual(s, sh)
            sh = s
        self.assertEqual(s, len(d))
        self.assertEqual(d[0]['eos'], eos_idx + idx)
        idx = 15
        set_sequence_start_idx(d, idx)
        s = 0
        sh = 0
        while s < len(d):
            s = d[s]['eos'] - idx
            self.assertNotEqual(s, sh)
            sh = s
        self.assertEqual(s, len(d))
        self.assertEqual(d[0]['eos'], eos_idx + idx)
        idx = 0
        set_sequence_start_idx(d, idx)
        s = 0
        sh = 0
        while s < len(d):
            s = d[s]['eos'] - idx
            self.assertNotEqual(s, sh)
            sh = s
        self.assertEqual(s, len(d))
        self.assertEqual(d[0]['eos'], eos_idx + idx)
        with self.assertRaises(AssertionError):
            set_sequence_start_idx(d, -10)

    def test_cv_splits(self):
        pass


class TestFtEx(TestCase):

    def setUp(self):
        self.data_str = '''The\tD
quick\tA
fox\tN
jumped\tV
across\tR
the\tD
river\tN
.\t.

The\tD
stupid\tA
wolf\tN
fell\tV
in\tI
the\tD
trap\tN
.\t.'''
        self.dp = 'tmp/data.%s.tmp' % time.asctime()
        open(self.dp, 'w').write(self.data_str)
        self.data = parse_tsv(self.dp, ('form', 'postag'))

    def tearDown(self):
        os.remove(self.dp)

    @staticmethod
    def fakeres(data, i, rel=0, b=None, p=None, *args, **kwargs):
        if b is None and p is None:
            return 'fakeres'
        try:
            v = b[data[i + rel]['form']][int(p)]
        except KeyError:
            v = None
        return 'fakeres[%s]=%s' % (rel, v)

    @staticmethod
    def winfakeres(fn, fw, fp, *args, **kwargs):
        prms = tuple() if fp is None else tuple(fp)
        for i in fw:
            yield (fn, i) + prms

    def test_parse_range(self):
        rr = [1, 2, 3, 10]
        rs = '1:3,10'
        r = parse_range(rs)
        self.assertSequenceEqual(r, rr)

    def test_ftvec_templ(self):

        fr = {'fox': range(10), 'wolf': range(10)}
        fr_dict = {'fakeres': fr}

        ftt_real = FeatureTemplate(fnx=[self.fakeres])
        ftt_real.add_win_features('word', [-3, -2, -1, 0, 1, 4], ())
        ftt_real.add_win_features('pos', [-2, 0], ())
        ftt_real.add_win_features('fakeres', [0], (fr, '0'))
        ftt_real.add_feature('fakeres')

        ftvex_spaceing = [
            '\tword:[-3:1,4];pos :   [-2,0];fakeres:[0],0;fakeres\t',
            'word:[-3:1,4];pos:[-2,0] ;fakeres:[0] , 0; fakeres',
            'word:[-3:1,4] ; pos:[-2,0] ; fakeres:[0],0;fakeres',
            'word:[ -3 : 1 , 4 ];pos:[-2,0];fakeres:[0],0;fakeres',
            'word:[-3:1,4];pos : [-2,0];fakeres : [0],0;fakeres',
            ' word:[-3:1,4];pos:[-2,0];fakeres:[0],0 ;fakeres',
            '    word :[-3:1,4];   pos :   [-2,0];fakeres:[0],0 ;fakeres',
            'word:[-3:1,4];pos:[-2,0];fakeres:[0],  0 ;fakeres',
            'word:[-3:1,4];pos:[-2,0];fakeres:[0],  0; ;fakeres'
        ]

        for ftvec in ftvex_spaceing:
            ftt = FeatureTemplate()
            ftt.parse_ftvec_templ(ftvec, fr_dict)
            self.assertSequenceEqual(ftt.vec, ftt_real.vec)

        ftt_real.fakeres(self.data, 0, *ftt_real.vec[-1][1:])

    def test_ftt_constructor(self):
        ftt = FeatureTemplate(fnx=[self.fakeres], win_fnx=[self.winfakeres])
        self.assertEqual(ftt.vec, [])
        self.assertIn(self.fakeres, ftt.__dict__.values())
        self.assertIn(self.winfakeres, ftt.win_fnx.values())

    def test_make_features(self):
        ftt = FeatureTemplate(fnx=[self.fakeres])

        fr = {'fox': range(10), 'wolf': range(10)}

        ftt.add_win_features('word', [-3, -2, 4], ())
        ftt.add_win_features('pos', [-2, 0], ())
        ftt.add_win_features('fakeres', [0], (fr, '0'))

        real_fts = [
            [
                ['The', 'w[-3]=None', 'w[-2]=None', 'w[4]=across', 'p[-2]=None',
                 'p[0]=D', 'fakeres[0]=None'],
                ['quick', 'w[-3]=None', 'w[-2]=None', 'w[4]=the', 'p[-2]=None',
                 'p[0]=A', 'fakeres[0]=None'],
                ['fox', 'w[-3]=None', 'w[-2]=The', 'w[4]=river', 'p[-2]=D',
                 'p[0]=N', 'fakeres[0]=0'],
                ['jumped', 'w[-3]=The', 'w[-2]=quick', 'w[4]=.', 'p[-2]=A',
                 'p[0]=V', 'fakeres[0]=None'],
                ['across', 'w[-3]=quick', 'w[-2]=fox', 'w[4]=None', 'p[-2]=N',
                 'p[0]=R', 'fakeres[0]=None'],
                ['the', 'w[-3]=fox', 'w[-2]=jumped', 'w[4]=None', 'p[-2]=V',
                 'p[0]=D', 'fakeres[0]=None'],
                ['river', 'w[-3]=jumped', 'w[-2]=across', 'w[4]=None',
                 'p[-2]=R', 'p[0]=N', 'fakeres[0]=None'],
                ['.', 'w[-3]=across', 'w[-2]=the', 'w[4]=None', 'p[-2]=D',
                 'p[0]=.', 'fakeres[0]=None']
            ],
            [
                ['The', 'w[-3]=None', 'w[-2]=None', 'w[4]=in', 'p[-2]=None',
                 'p[0]=D', 'fakeres[0]=None'],
                ['stupid', 'w[-3]=None', 'w[-2]=None', 'w[4]=the', 'p[-2]=None',
                 'p[0]=A', 'fakeres[0]=None'],
                ['wolf', 'w[-3]=None', 'w[-2]=The', 'w[4]=trap', 'p[-2]=D',
                 'p[0]=N', 'fakeres[0]=0'],
                ['fell', 'w[-3]=The', 'w[-2]=stupid', 'w[4]=.', 'p[-2]=A',
                 'p[0]=V', 'fakeres[0]=None'],
                ['in', 'w[-3]=stupid', 'w[-2]=wolf', 'w[4]=None', 'p[-2]=N',
                 'p[0]=I', 'fakeres[0]=None'],
                ['the', 'w[-3]=wolf', 'w[-2]=fell', 'w[4]=None', 'p[-2]=V',
                 'p[0]=D', 'fakeres[0]=None'],
                ['trap', 'w[-3]=fell', 'w[-2]=in', 'w[4]=None', 'p[-2]=I',
                 'p[0]=N', 'fakeres[0]=None'],
                ['.', 'w[-3]=in', 'w[-2]=the', 'w[4]=None', 'p[-2]=D', 'p[0]=.',
                 'fakeres[0]=None']
            ]
        ]

        data = list(gsequences(self.data, cols=['form', 'postag']))

        for d, rfv in zip(data, real_fts):
            for i, rf in enumerate(rfv):
                fts = ftt.make_fts(d, i)
                self.assertItemsEqual(fts, rf)

    def test_word(self):
        for i in [-4, -1, 0, 2]:
            w = FeatureTemplate.word(self.data, 2, i)
            rel = 2 + i
            rw = 'w[%s]=%s' % (i, self.data[rel][0] if rel >= 0 else None)
            self.assertEqual(w, rw)

    def test_pos(self):
        for i in [-4, -1, 0, 2]:
            w = FeatureTemplate.pos(self.data, 2, i)
            rel = 2 + i
            rw = 'p[%s]=%s' % (i, self.data[rel][1] if rel >= 0 else None)
            self.assertEqual(w, rw)

    def test_chunk(self):
        dtn = self.data.dtype.names
        self.data.dtype.names = (dtn[0], 'chunktag',) + dtn[2:]
        for i in [-4, -1, 0, 2]:
            w = FeatureTemplate.chunk(self.data, 2, i)
            rel = 2 + i
            rw = 'ch[%s]=%s' % (i, self.data[rel][1] if rel >= 0 else None)
            self.assertEqual(w, rw)

    def test_can(self):
        for i in [-4, -1, 0, 2]:
            w = FeatureTemplate.can(self.data, 2, i)
            r = 2 + i
            rwv = re.sub('.',
                         'x',
                         self.data[r][0]) if len(self.data) > r >= 0 else None
            rw = 'can[%s]=%s' % (i, rwv)
            self.assertEqual(w, rw)

    def test_isnum(self):
        self.data[3][0] = '10'
        for i in [-4, -1, 0, 2]:
            w = FeatureTemplate.isnum(self.data, 2, i)
            r = 2 + i
            rwv = i == 3 if len(self.data) > r >= 0 else None
            rw = 'isnum[%s]=%s' % (i, rwv)
            self.assertEqual(w, rw)

    def test_emb(self):
        e = {'fox': range(100), 'quick': list(reversed(range(100)))}
        i = 2
        for rel in [-4, -1, 0, 2]:
            for j in [4, 6, 8]:
                w = FeatureTemplate.emb(self.data, i, rel, j, e)
                r = i + rel
                v = None
                if len(self.data) > r >= 0:
                    v = e.get(self.data[r][0], [None for _ in range(100)])[j]
                rw = 'emb[%s][%s]=%s' % (rel, j, v)
                self.assertEqual(w, rw)

    def test_cls(self):
        c = {'fox': 1, 'quick': 2}
        i = 2
        for rel in [-4, -1, 0, 2]:
            w = FeatureTemplate.cls(self.data, i, rel, c)
            r = i + rel
            v = None
            if len(self.data) > r >= 0:
                v = c.get(self.data[r][0], None)
            rw = 'cnum[%s]=%s' % (rel, v)
            self.assertEqual(w, rw)

    def test_brown(self):
        b = {'fox': '00011110011010', 'quick': '11001001110011'}
        i = 2
        p = 10
        for rel in [-4, -1, 0, 2]:
            w = FeatureTemplate.brown(self.data, i, rel, b, p)
            r = i + rel
            v = None
            if len(self.data) > r >= 0:
                v = b.get(self.data[r][0], None)
            if v:
                v = v[:10]
            rw = 'cn[%s]:%s=%s' % (rel, p, v)
            self.assertEqual(w, rw)

    def test_suff(self):
        b = {'ox', 'ick', 'across'}
        i = 2
        p = 5
        for rel in [-4, -1, 0, 2]:
            w = FeatureTemplate.suff(self.data, i, rel, b, p)
            r = i + rel
            v = None
            if len(self.data) > r >= 0:
                if self.data[r]['form'].endswith('ox'):
                    v = 'ox'
                elif self.data[r]['form'].endswith('ick'):
                    v = 'ick'
            rw = 'sfx[%s]=%s' % (rel, v)
            self.assertEqual(w, rw)

    def test_pref(self):
        b = {'fo', 'qui', 'across'}
        i = 2
        p = 5
        for rel in [-4, -1, 0, 2]:
            w = FeatureTemplate.pref(self.data, i, rel, b, p)
            r = i + rel
            v = None
            if len(self.data) > r >= 0:
                if self.data[r]['form'].startswith('fo'):
                    v = 'fo'
                elif self.data[r]['form'].startswith('qui'):
                    v = 'qui'
            rw = 'sfx[%s]=%s' % (rel, v)
            self.assertEqual(w, rw)

    def test_nrange(self):
        rng = nrange(0, 10, 3)
        self.assertSequenceEqual(rng, range(0, 9, 1))
        rng = nrange(0, 11, 3)
        self.assertSequenceEqual(rng, range(0, 10, 1))
        rng = nrange(0, 12, 3)
        self.assertSequenceEqual(rng, range(0, 11, 1))
        rng = nrange(0, 13, 1)
        self.assertSequenceEqual(rng, range(0, 14, 1))
        rng = nrange(0, 14, 2)
        self.assertSequenceEqual(rng, range(0, 14, 1))

    def test_parse_ng_range(self):
        nrng = parse_ng_range(range(10), 3)
        self.assertSequenceEqual(nrng, range(8))
        nrng = parse_ng_range(range(10) + range(12, 15), 3)
        self.assertSequenceEqual(nrng, range(8) + range(12, 15, 3))
        nrng = parse_ng_range(range(10) + range(12, 16) + range(20, 22), 2)
        self.assertSequenceEqual(nrng, range(9) + range(12, 15) + range(20, 21))

    def test_nword(self):
        ft = FeatureTemplate.nword(self.data[:8], 3, 2, 2)
        self.assertEqual(ft, '2w[2]=theriver')
        ft = FeatureTemplate.nword(self.data[:8], 5, 2, 2)
        self.assertEqual(ft, '2w[2]=None')
        ft = FeatureTemplate.nword(self.data[:8], 5, -1, 2)
        self.assertEqual(ft, '2w[-1]=acrossthe')
        ft = FeatureTemplate.nword(self.data[:8], 1, -2, 3)
        self.assertEqual(ft, '3w[-2]=None')

    def test_npos(self):
        ft = FeatureTemplate.npos(self.data[:8], 3, 2, 2)
        self.assertEqual(ft, '2p[2]=DN')
        ft = FeatureTemplate.npos(self.data[:8], 5, 2, 2)
        self.assertEqual(ft, '2p[2]=None')
        ft = FeatureTemplate.npos(self.data[:8], 5, -1, 2)
        self.assertEqual(ft, '2p[-1]=RD')
        ft = FeatureTemplate.npos(self.data[:8], 1, -2, 3)
        self.assertEqual(ft, '3p[-2]=None')


class TestEval(TestCase):

    def test_pos(self):
        data = [{'postag': 'N' if x % 2 else 'V',
                 'guesstag': 'N' if x % 2 else 'V'} for x in range(10)]
        data[0]['guesstag'] = 'N'
        r = pos(data)
        self.assertEqual(r['Total']['accuracy'], 0.9)
        self.assertEqual(r['V']['accuracy'], 0.8)
        self.assertEqual(r['N']['accuracy'], 1.0)

    def test_conll(self):
        """This tests only the data is correctly processed and parsed. It does
        not test the CoNLL evaluation script results.


        """
        dt = 'a10,a10,a10,a10,int32'
        data = np.zeros(10, dtype=dt)
        data.dtype.names = ['form', 'postag', 'chunktag', 'guesstag', 'eos']
        for x in range(10):
            data[x] = ('bla', 'N', 'B-NP' if x % 2 else 'B-VP',
                       'B-NP' if x % 2 else 'B-VP', 9 if x == 0 else -1)
        data[0]['guesstag'] = 'B-NP'
        data[0]['guesstag'] = 'I-NP'
        r = conll_old(data)
        self.assertAlmostEqual(float(r['Total']['fscore']), 90.0)
        self.assertAlmostEqual(float(r['VP']['fscore']), 88.89)
        self.assertAlmostEqual(float(r['NP']['fscore']), 90.91)


class TestTagger(TestCase):
    pass