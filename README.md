CRFSuiteTagger
==============

_CRFSuiteTagger_ is a sequence tagger based on the [pycrfsuite](https://github.com/tpeng/python-crfsuite "pycrfsuite") python wrapper for [CRFSuite](http://www.chokkan.org/software/crfsuite/ "CRFSuite"). It is built for chunking, NER, and other BIO (also referred to as IOB) based text annotation tasks.

### Why would you need this?

_CRFSuiteTagger_ has a wide selection of common features, and the capability to easily integrate additional ones. The features are controlled using a simple string-based feature template. Additional features can be easily added through new _feature generating functions_ (see `crfsuitetagger.ftex`) passed on the `CRFSuiteTagger` constructor.

### Installation

You should be able to install _CRFSuiteTagger_ as any other Python package:

    python setup.py install

### Dependencies

You will need the following Python packages and one of my other libraries:

* [pycrfsuite](https://github.com/tpeng/python-crfsuite  "pycrfsuite") - python wrapper for CRFSuite
* [numpy](http://www.numpy.org/ "NumPy") - you should it
* [bioeval](https://github.com/savkov/bioeval "bioeval") - my library for evaluating BIO style annotation, which replaces the perl script from [CoNLL-2000](http://ilk.uvt.nl/team/sabine/chunklink/chunklink_2-2-2000_for_conll.pl)

### TODO

* command line interface
* migrate data structure to [pandas](http://pandas.pydata.org/ "pandas")
* more examples

### See Also

If you are interested in other sequence taggers, you might want to look at:

* [Stanford NLP](http://nlp.stanford.edu/software/lex-parser.shtml) -- POS tagger
* [ARK](http://www.ark.cs.cmu.edu/TweetNLP/) -- POS tagger for tweets
* [YamCha](http://chasen.org/~taku/software/yamcha/) -- BIO tagger/chunker
* [CRF++](http://taku910.github.io/crfpp/) -- BIO tagger/chunker
* [Wapiti](https://wapiti.limsi.fr/) -- POS & BIO tagger/chunker