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


# Use this script to download the CoNLL-2000 data files in this directory.
# Make sure you run this script from the directory it was placed in.

import os, gzip, wget


def extract(fp):
    f = gzip.open(fp, 'rb')
    with open(fp[:-3], 'w') as fh:
        fh.write(f.read())
    f.close()

try:
    os.makedirs('../tmp')
except OSError:
    pass

try:
    os.makedirs('thesauri')
except OSError:
    pass

train_url = 'http://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz'
test_url = 'http://www.cnts.ua.ac.be/conll2000/chunking/test.txt.gz'

# The Turian embeddings were available from the web site of MetaOptimize,
# which is currently unavailable.
#
# turian_url = 'http://pylearn.org/turian/brown-clusters/brown-rcv1.clean.\
#               tokenized-CoNLL03.txt-c1000-freq1.txt'
# turian_emb_url =

stanford_clusters_url = \
    'http://nlp.stanford.edu/software/egw4-reut.512.clusters'

train = wget.download(train_url)
test = wget.download(test_url)

# os.remove('thesauri/brown-rcv1.clean.tokenized-CoNLL03.txt-c1000-freq1.txt')
# turian = wget.download(
#     turian_url,
#     out='thesauri/brown-rcv1.clean.tokenized-CoNLL03.txt-c1000-freq1.txt'
# )
# os.remove('thesauri/embeddings-scaled.EMBEDDING_SIZE=50.txt')
# turian_emb = wget.download(
#     turian_emb_url,
#     out='thesauri/embeddings-scaled.EMBEDDING_SIZE=50.txt'
# )

os.remove('thesauri/egw4-reut.512.clusters')
stanford = wget.download(
    stanford_clusters_url,
    out='thesauri/egw4-reut.512.clusters'
)


extract(train)
extract(test)

os.remove(train)
os.remove(test)