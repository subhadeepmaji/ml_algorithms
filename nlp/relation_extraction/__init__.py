__author__ = 'subhadeepmaji'

MODEL_PATH = "nlp/relation_extraction/data_sink/models"
ELASTIC_HOST = 'localhost'

from util import LoggerConfig
import logging

config = dict(LoggerConfig.logger_config)
config['filename'] = 'relation_extraction.log'

logging.basicConfig(**config)
