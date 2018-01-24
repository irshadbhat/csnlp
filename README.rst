Stack-augmented Neural Dependency Parser for Code Switching texts
=================================================================

Neural Stacking Dependency Parsers for monolingual, multilingual and code-switching data.

Install dependencies
^^^^^^^^^^^^^^^^^^^^

  - pip install -r requirements.txt
  - `DyNet version 2.0
    <https://github.com/clab/dynet>`_

Files you care about?
^^^^^^^^^^^^^^^^^^^^^

  - ``mono_tagger.py`` Monolingual Tagger: Use this file to train a monolingual POS-tagger or NER-tagger. The training file should be in TNT format, i.e. each line should contain a word and a tag seperated by a tab and the sentences seperated by an empty line. You can directly pass the CONLL file as well. In case of CONLL input, if you set the UD flag (check ``python mono_tagger.py --help``) the code takes 4th column as tag which is the POS-tag column for Universal Dependencies. If the UD flag is not set, the code takes 5th column as tag. See ``python mono_tagger.py --help`` for various command-line options and network hyperparameters.
  - ``snn_mono_tagger.py`` Neural Stacking Monolingual Tagger: Train a stacking tagger on top of the base model trained using ``mono_tagger.py``. Why stacking and where? Suppose we want to train a POS tagger for entertainment domain, but we have lesser data samples of entertainment domain compared to newswire. We can add the two datasets and train the model on the augmented data or we can use neural stacking. In case of stacking, we train a base model using the bigger dataset/domain and then train the stacking model using the smaller dataset/domain. Stacking models take features from the base model using neural stacking. Stacking models seem to outperform the models trained on augmented data. See ``python snn_mono_tagger.py --help`` for various command-line options and network hyperparameters.
  - ``mono_jm_parser.py`` Monolingual Parser: Use this file to train a monolingual parser. The parser jointly learns the POS tags and dependency relations. The training file should be in CONLL format. See ``python mono_jm_parser.py --help`` for various command-line options and network hyperparameters.
  - ``snn_mono_jm_parser.py`` Neural Stacking Monolingual Parser: Train a stacking parser on top of the base model trained using ``mono_jm_parser.py``. See ``python snn_mono_jm_parser.py --help`` for various command-line options and network hyperparameters.
  - ``polyglot_tagger.py``  Multilingual Tagger: 
  - ``snn_polyglot_tagger.py`` Neural Stacking Multilingual Tagger:
  - ``polyglot_jm_parser.py`` Multilingual Parser:
  - ``snn_polyglot_jm_parser.py`` Neural Stacking Multilingual Parser:
