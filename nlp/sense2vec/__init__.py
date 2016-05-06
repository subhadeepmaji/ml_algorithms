__author__ = "subhadeepmaji"

import logging
from util import LoggerConfig

config = dict(LoggerConfig.logger_config)
config['filename'] = 'sense2vec.log'

CONJUNCTION = {'CC', 'IN'}
CARDINAL = {'CD'}
ADJECTIVE = {'JJ', 'JJR', 'JJS'}
NOUN = {'NN', 'NNS', 'NNP', 'NNPS'}
VERB = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
SYMBOL = {'SYM'}

logging.basicConfig(**config)
