from mongoengine import connect
from models      import NewsArticle as NA

MONGO_HOST = "localhost"
MONGO_PORT = 27017
DB_NAME    = "news"
ALIAS_NAME = "news"

def build_connections(db_name, alias_name, connect_str):
    if connect_str:
        db_connection = connect(db = db_name, alias = alias_name,
                                host = connect_str)
    else:
        host = MONGO_HOST
        port = MONGO_PORT
        db_connection = connect(db = db_name, alias = alias_name,
                                host = host, port = port)
        
class NewsArticleDao(object):

    def __init__(self, db_name  = DB_NAME, alias_name = ALIAS_NAME,
                 connect_str = None):

        self.db_name = db_name
        self.alias_name = alias_name
        self.connect_str = connect_str
        build_connections(self.db_name, self.alias_name, self.connect_str)

    def form(self, source, title, location, text, summary = None,
             keywords = [], open_graph = None, publish_date = None):

        article = NA.NewsArticle(source = source, title = title,
                                 location = location, text = text,
                                 summary = summary, keywords = keywords,
                                 open_graph = open_graph,
                                 publish_date = publish_date)
        return article

    def save(self, article):
        article.save()

    def form_and_save(self, *args, **kwargs):
        article = self.form(*args, **kwargs)
        self.save(article)
    

