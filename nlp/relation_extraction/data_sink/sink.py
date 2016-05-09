import importlib
import logging
import re
from collections import namedtuple
from os import listdir
from os.path import isfile, join, abspath

from concurrent.futures import ThreadPoolExecutor
from elasticsearch_dsl.connections import connections
from queue import Full, Empty, Queue

from nlp.relation_extraction.data_sink import ELASTIC_HOST,ELASTIC_PORT
from nlp.relation_extraction.data_sink import MODEL_PATH

logger = logging.getLogger(__name__)
PY_FILE_REGEX = re.compile(".*\.py$")
ModelIdentifier = namedtuple("ModelIdentifier", ['index', 'mapping', 'model_class'])


class SinkLoader:

    def __init__(self):
        self.model_location = abspath(MODEL_PATH)
        self.load_path = ".".join(MODEL_PATH.split("/"))
        self.models = list()
        self.data_sinks = dict()

    def form_sinks(self):
        model_modules = [c for c in listdir(self.model_location) if
                         isfile(join(self.model_location, c)) if c != '__init__.py']

        model_modules = [m for m in model_modules if PY_FILE_REGEX.match(m)]
        for model_module in model_modules:
            # get the name of the class
            model_module = model_module.split('.')[0]
            try:
                module_path = self.load_path + '.' + model_module

                model_class = model_module
                module = importlib.import_module(module_path)
                self.models.append(ModelIdentifier(index=module.index, mapping=module.mapping,
                                                   model_class=module.model_class))
            except (ImportError, Exception) as e:
                raise RuntimeError("Error importing the module ", e)

        for model in self.models:
            model_name = model.index + "." + model.mapping
            connections.create_connection(model_name, hosts=[ELASTIC_HOST], port=ELASTIC_PORT)
            data_sink = ElasticDataSink(model_name, connections.get_connection(model_name), model)
            self.data_sinks[model_name] = data_sink

    def close_sinks(self):
        for data_sink_name in self.data_sinks.keys():
            connections.remove_connection(data_sink_name)


class ElasticDataSink:

    def __init__(self, name, conn, model_identifier, workers=5, bound=10000):
        self.name = name
        self.conn = conn
        self.model_identifier = model_identifier
        self.queue = Queue(maxsize=bound)
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def start(self):
        self.model_identifier.model_class.init(using=self.conn)

    def __sink_item(self):
        try:
            item = self.queue.get_nowait()
            save_status = item.save(using=self.conn)
            if not save_status:
                logger.error("Error saving the item to the sink")
            else:
                logger.info("item saved to the sink")
        except Empty as e:
            logger.warn("sink queue is empty")
            logger.warn(e)

    def sink_item(self, item):
        assert isinstance(item, self.model_identifier.model_class), \
            " item must be instance of " + self.model_identifier.model_class

        try:
            self.queue.put(item, timeout=10)
            self.pool.submit(self.__sink_item)
        except Full as e:
            logger.error("sink queue is full")
            logger.error(e)



