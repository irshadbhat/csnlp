Neural Stacking Dependency Parsers for Code Switching texts
===========================================================

Neural Stacking Dependency Parsers for monolingual, multilingual and code switching data. This repository contains the source code described in our paper `Universal Dependency Parsing for Hindi-English Code-switching`_.

.. _`Universal Dependency Parsing for Hindi-English Code-switching`:

Install dependencies
^^^^^^^^^^^^^^^^^^^^

  - pip install -r requirements.txt
  - `DyNet version 2.0
    <https://github.com/clab/dynet>`_

Files you care about?
^^^^^^^^^^^^^^^^^^^^^

  - ``mono_tagger.py`` Monolingual Tagger: Use this file to train a monolingual POS-tagger or NER-tagger. The training file should be in TNT format, i.e. each line should contain a word and a tag separated by a tab and the sentences separated by an empty line. You can directly pass the CONLL file as well. In case of CONLL input, if you set the UD flag (check ``python mono_tagger.py --help``) the code takes 4th column as tag which is the POS-tag column for Universal Dependencies. If the UD flag is not set, the code takes 5th column as tag. See ``python mono_tagger.py --help`` for various command-line options and network hyperparameters.
  - ``snn_mono_tagger.py`` Neural Stacking Monolingual Tagger: Train a stacking tagger on top of the base model trained using ``mono_tagger.py``. Why stacking and where? Suppose we want to train a POS tagger for entertainment domain, but we have lesser data samples of entertainment domain compared to newswire. We can add the two datasets and train the model on the augmented data or we can use neural stacking. In case of stacking, we train a base model using the bigger dataset/domain and then train the stacking model using the smaller dataset/domain. Stacking models take features from the base model using neural stacking. Stacking models seem to outperform the models trained on augmented data. See ``python snn_mono_tagger.py --help`` for various command-line options and network hyperparameters.
  - ``mono_jm_parser.py`` Monolingual Parser: Use this file to train a monolingual parser. The parser jointly learns the POS tags and dependency relations. The training file should be in CONLL format. See ``python mono_jm_parser.py --help`` for various command-line options and network hyperparameters.
  - ``snn_mono_jm_parser.py`` Neural Stacking Monolingual Parser: Train a stacking parser on top of the base model trained using ``mono_jm_parser.py``. See ``python snn_mono_jm_parser.py --help`` for various command-line options and network hyperparameters.
  - ``polyglot_tagger.py``  Multilingual Tagger: Use this code to train a POS tagger for code switching data using only the monolingual datasets. The training file should be in TNT or CONLL format. See ``python polyglot_tagger.py --help`` for various command-line options and network hyperparameters.
  - ``snn_polyglot_tagger.py`` Neural Stacking Multilingual Tagger: Train a stacking tagger (using code switching data) on top of the base model trained using ``polyglot_tagger.py``. The code switching training file should be in tri-column (normalized/back-transliterated word, POS tag, lang tag) or CONLL format. In case of CONLL format, the normalized/back-transliterated words should be in 3rd column and the language tags should be in 9th column. See ``python snn_polyglot_tagger.py --help`` for various command-line options and network hyperparameters. 
  - ``polyglot_jm_parser.py`` Multilingual Parser: Use this code to train a parser for code switching data using only the monolingual treebanks. See ``python polyglot_jm_parser.py --help`` for various command-line options and network hyperparameters.
  - ``snn_polyglot_jm_parser.py`` Neural Stacking Multilingual Parser: Train a stacking parser (using code switching data) on top of the base model trained using ``polyglot_jm_parser.py``. The normalized/back-transliterated words should be in 3rd column and the language tags should be in 9th column. See ``python snn_polyglot_jm_parser.py --help`` for various command-line options and network hyperparameters. 


Training Models
^^^^^^^^^^^^^^^

.. parsed-literal::

  python mono_jm_parser.py --help
  
    --dynet-seed SEED
    --train TRAIN [TRAIN ...]  CONLL Train file
    --dev DEV [DEV ...]        CONLL Dev/Test file
    --test TEST                Raw Test file
    --pretrained-embds EMBD    Pretrained word2vec Embeddings
    --elimit ELIMIT            load only top-n pretrained word vectors (default = all vectors)
    --lang LANG                3-letter ISO language code e.g., eng for English, hin for Hindi
    --trainer TRAINER          Trainer [momsgd|adam|adadelta|adagrad]
    --activation-fn ACT_FN     Activation function [tanh|rectify|logistic]
    --ud UD                    1 if UD treebank else 0
    --iter ITER                No. of Epochs
    --bvec BVEC                1 if binary embedding file else 0
    --save-model SAVE_MODEL    Specify path to save model
    --load-model LOAD_MODEL    Load Pretrained Model
    --output-file OUTFILE      Output File
    --daemonize                Daemonize parser
    --port DAEMONPORT          Specify a port number

  python --dynet-seed 127 --train /path/to/train-file --dev /path/to/dev-file --pretrained-embds 
         /path/to/gensim-pretrained-embedding --elimit 300000 --lang eng --trainer adam --ud 1 
         --iter 100 --bvec 1 --save /path/to/save-model


Testing Models
^^^^^^^^^^^^^^

You can test the models in four different settings:

1) Annotated Test/Dev file
##########################

::

    python mono_jm_parser.py --load /path/to/saved-model --dev /path/to/conll-test-or-dev-file

2) Raw Test file
################

::

    python mono_jm_parser.py --load /path/to/saved-model --test /path/to/raw-text-file

3) Call within Python
#####################

.. code:: python

    >>> from mono_jm_parser import *
    [dynet] random seed: 497379357
    [dynet] allocating memory: 512MB
    [dynet] memory allocation done.
    >>> 
    >>> parser = Parser(model='/home/irshad/Projects/BITProjects/nsdp-cs-models/PTB/PARSER/en-ptb-parser')
    >>> raw_sent = 'Give me back my peace of mind .'.split()
    >>> 
    >>> print parse_sent(parser, raw_sent)
    1	Give	_	VB	_	_	0	root	_	_
    2	me	_	PRP	_	_	1	iobj	_	_
    3	back	_	RP	_	_	1	prt	_	_
    4	my	_	PRP$	_	_	5	poss	_	_
    5	peace	_	NN	_	_	1	dobj	_	_
    6	of	_	IN	_	_	5	prep	_	_
    7	mind	_	NN	_	_	6	pobj	_	_
    8	.	_	.	_	_	1	punct	_	_
    >>> 

4) Daemonize
############

Run the parser in daemonize mode:

.. parsed-literal::

    python mono_jm_parser.py --load ~/Projects/BITProjects/nsdp-cs-models/PTB/PARSER/en-ptb-parser --daemonize --port 4000
    [dynet] random seed: 2719235480
    [dynet] allocating memory: 512MB
    [dynet] memory allocation done.
    Loading Models ...
    Done!

Open a new terminal and parse sentences using the command:

.. parsed-literal::

    echo 'I see skies of blue , clouds of white , bright blessed days , dark sacred night .' | nc localhost 4000
    1	I	_	PRP	_	_	2	nsubj	_	_
    2	see	_	VBP	_	_	0	root	_	_
    3	skies	_	NNS	_	_	2	dobj	_	_
    4	of	_	IN	_	_	3	prep	_	_
    5	blue	_	JJ	_	_	7	amod	_	_
    6	,	_	,	_	_	7	punct	_	_
    7	clouds	_	NNS	_	_	3	conj	_	_
    8	of	_	IN	_	_	7	prep	_	_
    9	white	_	JJ	_	_	13	amod	_	_
    10	,	_	,	_	_	13	punct	_	_
    11	bright	_	RB	_	_	12	advmod	_	_
    12	blessed	_	JJ	_	_	13	amod	_	_
    13	days	_	NNS	_	_	8	pobj	_	_
    14	,	_	,	_	_	7	punct	_	_
    15	dark	_	JJ	_	_	17	amod	_	_
    16	sacred	_	JJ	_	_	17	amod	_	_
    17	night	_	NN	_	_	7	npadvmod    _	_
    18	.	_	.	_	_	2	punct	_	_

Use Pretrained Models
^^^^^^^^^^^^^^^^^^^^^

You can use pretrained models from `nsdp-cs-models <https://bitbucket.org/irshadbhat/nsdp-cs-models>`_.
