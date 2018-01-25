Neural Stacking Dependency Parsers for Code Switching texts
===========================================================

Neural Stacking Dependency Parsers for monolingual, multilingual and code switching data.

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

Testing Models
^^^^^^^^^^^^^^^


Use Pretrained Models
^^^^^^^^^^^^^^^^^^^^^

You can use pretrained models from `nsdp-cs-models <https://bitbucket.org/irshadbhat/nsdp-cs-models>`_.
