__author__ = "subhadeepmaji"

import logging
from util import LoggerConfig

config = dict(LoggerConfig.logger_config)
config['filename'] = 'query_understanding.log'

logging.basicConfig(**config)
