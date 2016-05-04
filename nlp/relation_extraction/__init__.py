__author__ = 'subhadeepmaji'

MODEL_PATH = "nlp/relation_extraction/data_sink/models"
ELASTIC_HOST = 'localhost'

from util import LoggerConfig
from collections import namedtuple
import logging, types, copy_reg


config = dict(LoggerConfig.logger_config)
config['filename'] = 'relation_extraction.log'

logging.basicConfig(**config)
JOB_LIB_TEMP_FOLDER = '.'

# Named tuple definitions of semantic role labeling
RelationTuple = namedtuple("RelationTuple", ['left_entity', 'right_entity', 'relation', 'sentence'])
RelationArgument = namedtuple('RelationArgument', ['A0', 'A1', 'A2', 'A3'])
RelationModifier = namedtuple('RelationModifier', ['DIR', 'LOC', 'TMP', 'MNR', 'EXT', 'PNC', 'CAU', 'NEG'])
EntityTuple = namedtuple('EntityTuple', ['index', 'value'])

POS_TAG_ENTITY = ['NN', 'PRP', 'PRP$', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS',
                  'CC', 'CD', 'IN', 'RB', 'RBR', 'RBS']
PRONOUN_PHRASES = ['S-PP', 'B-PP', 'I-PP', 'E-PP']

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

