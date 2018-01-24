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

  - ``mono_tagger.py`` Monolingual Tagger: Use this file to train a monolingual POS-tagger or NER-tagger. The training file should be in TNT format, i.e. each line should contain a word and a tag seperated by a tab and the sentences seperated by an empty line. You can also pass the CONNL file. In case of CONLL input, if you set the UD flag (check ``python mono_tagger.py --help``) the code takes 4th column as tag which is the POS-tag column for Universal Dependencies. If the UD flag is not set, the code takes 5th column as tag.
  - ``snn_mono_tagger.py`` Neural Stacking Monolingual Tagger: 
  - ``mono_jm_parser.py`` Monolingual Parser:
  - ``snn_mono_jm_parser.py`` Neural Stacking Monolingual Parser:
  - ``polyglot_tagger.py``  Multilingual Tagger:
  - ``snn_polyglot_tagger.py`` Neural Stacking Multilingual Tagger:
  - ``polyglot_jm_parser.py`` Multilingual Parser:
  - ``snn_polyglot_jm_parser.py`` Neural Stacking Multilingual Parser:
