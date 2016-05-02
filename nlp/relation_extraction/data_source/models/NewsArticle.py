from mongoengine import Document
from mongoengine import StringField

db_name = 'news'
db_alias = 'news'
collection_name = 'news_article'
relation_fields = ['text']

class NewsArticle(Document):
    source = StringField(required=True)
    title = StringField(required=True)
    text = StringField(required=True)

    meta = {
        'db_alias' : db_alias,
        'collection' : collection_name,
        'strict' : False
    }

model_class = NewsArticle

