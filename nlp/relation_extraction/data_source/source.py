from nlp.relation_extraction.data_source import MODEL_PATH
from nlp.relation_extraction.data_source import MONGO_HOST, MONGO_PORT

from mongoengine import connect
from os import listdir
from os.path import isfile, join, abspath
from collections import namedtuple
from queue import Queue,Empty
from enum import Enum
from threading import Thread, Lock

from concurrent.futures import ThreadPoolExecutor
import importlib, re, logging

logger = logging.getLogger(__name__)

PY_FILE_REGEX = re.compile(".*\.py$")
ModelIdentifier = namedtuple('ModelIdentifier', ['db_name', 'db_alias', 'collection_name',
                                                 'fields', 'model_class'])


class SourceLoader:
    """
    Relation Extraction Data Source
    """
    def __init__(self, connect_str=None):
        self.model_location = abspath(MODEL_PATH)
        self.load_path = ".".join(MODEL_PATH.split("/"))
        self.connect_str = connect_str
        self.models = list()
        self.data_sources = dict()

    def form_sources(self):
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
                self.models.append(ModelIdentifier(db_name=module.db_name, db_alias=module.db_alias,
                                                   collection_name=module.collection_name,
                                                   fields=module.relation_fields,
                                                   model_class=module.model_class))
            except (ImportError, Exception) as e:
                raise RuntimeError("Error importing the module ", e)

        for model in self.models:
            if self.connect_str:
                conn = connect(model.db_name, model.db_alias, host=self.connect_str)
            else:
                conn = connect(model.db_name, model.db_alias, host=MONGO_HOST, port=MONGO_PORT)
            model_name = model.db_name + "." + model.db_alias + "." + model.collection_name
            data_source = MongoDataSource(model_name, conn, model)
            self.data_sources[model_name] = data_source

    def close_sources(self):
        for data_source in self.data_sources.values():
            data_source.conn.close()


class MongoDataSource:
    """
    Mongo Data Source should be used in a with clause
    # TO DO : Enhance to allow filters in the query, also include support
    for cursor managers to kill cursors on demand and misc features
    to handle cursor timeouts, batch read on cursor etc.
    """
    class SourceState(Enum):
        inited = 1
        started = 2
        stopped = 3

    def __init__(self, name, db_conn, model_identifer, workers=5, bound=10000):
        self.name = name
        self.db_conn = db_conn
        self.model_identifier = model_identifer
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.state = MongoDataSource.SourceState.inited
        self.queue = Queue(maxsize=bound)
        self.jobs = set()
        self.item_lock = Lock()

    def start(self, query_filter=None):
        """
        start a data source object
        :return:
        """
        assert self.state == MongoDataSource.SourceState.inited, \
            "Data source not in inited state, cant start"

        self.state = MongoDataSource.SourceState.started
        self.query_filter = query_filter
        self.items = self.model_identifier.model_class.objects

    def stop(self):
        """
        stop a datasource
        :return:
        """
        if self.state == MongoDataSource.SourceState.stopped:
            # already stopped return gracefully, without doing anything
            return
        self.state = MongoDataSource.SourceState.stopped

        for job in list(self.jobs):
            job.cancel()

        # shutdown the executor pool
        self.pool.shutdown(wait=True)

    def __enqueue_item(self, item_future):
        self.jobs.remove(item_future)
        if self.state != MongoDataSource.SourceState.started :
            # dont do anything just return, if the source is not in correct state
            return

        if item_future.exception():
            logger.error("Error evaluating the future")
            logger.error(item_future.exception())
            self.stop()
        else:
            self.queue.put(item_future.result(), timeout=10)

    def __submit_read_task(self):
        kwargs = {'query_filter': self.query_filter}

        # add this job only if the source is in started state
        if self.state == MongoDataSource.SourceState.started:
            f = self.pool.submit(self._get_item, **kwargs)
            self.jobs.add(f)
            f.add_done_callback(self.__enqueue_item)

    def get_item(self):
        assert self.state == MongoDataSource.SourceState.started, \
            "data source should be in started state"
        try:
            if self.queue.qsize() < 10:
                for _ in xrange(30): self.__submit_read_task()

            item = self.queue.get(block=True, timeout=10)
            self.queue.task_done()
            return item

        except Empty as e:
            logger.error("empty queue ")
            logger.error(e)
            return None

        except Exception as e:
            raise e

    def _get_item(self, query_filter=None):
        """
        get a data item from the underlying source
        :param query_filter: a dict as filter option(s) on the query,
        default: None, retrieve all data in the collection
        :return:
        """
        try:
            self.item_lock.acquire()
            item = self.items.next()
            self.item_lock.release()

            data_field_values = []
            for f in self.model_identifier.fields:
                data_field_values.append(item[f])
            return tuple(data_field_values)

        except StopIteration as e:
            raise RuntimeError("No more elements in the source", e)

        except Exception as e:
            raise RuntimeError("Unknown error on Source", e)
