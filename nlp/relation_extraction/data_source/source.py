import importlib
import logging
import re
from collections import namedtuple
from os import listdir
from os.path import isfile, join, abspath

from mongoengine import connect

from nlp.relation_extraction.data_source import MODEL_PATH
from nlp.relation_extraction.data_source import MONGO_HOST, MONGO_PORT

logger = logging.getLogger(__name__)

PY_FILE_REGEX = re.compile(".*\.py$")
ModelIdentifier = namedtuple('ModelIdentifier', ['db_name', 'db_alias', 'collection_name',
                                                 'fields', 'payload_fields', 'function_fields',
                                                 'model_class'])


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
                                                   payload_fields = module.payload_fields,
                                                   function_fields = module.function_fields,
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
    def __init__(self, name, db_conn, model_identifer):
        self.name = name
        self.db_conn = db_conn
        self.model_identifier = model_identifer

    def start(self, query_filter=None):
        self.query_filter = query_filter
        self.items = self.model_identifier.model_class.objects

    def __iter__(self):
        return self

    def __getstate__(self):
        state = dict()
        return state

    def __setstate__(self, d):
        self.__dict__.update(d)

    def next(self):
        try:
            item = self.items.next()
            data_field_values = []

            for f in self.model_identifier.fields:
                field_value = item[f]
                if field_value[-1] not in ['.', '?']:
                    field_value += '.'
                data_field_values.append((f, field_value))

            if self.model_identifier.payload_fields:
                for f in self.model_identifier.payload_fields:
                    field_value = item[f]
                    data_field_values.append(("payload", field_value))

            if self.model_identifier.function_fields:
                for f in self.model_identifier.function_fields:
                    field_value = item[f]
                    field_value = field_value.split(" ")
                    i = [index for index, e in enumerate(field_value) if e.find(":") != -1]
                    if i and i[0] != -1:
                        field_value = " ".join(field_value[:i[0]])
                        data_field_values.append(("ff", field_value))

            return tuple(data_field_values)

        except StopIteration as e:
            raise e
        except Exception as e:
            raise RuntimeError("Unknown error on Source", e)
