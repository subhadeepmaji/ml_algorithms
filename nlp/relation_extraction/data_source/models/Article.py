from mongoengine import Document
from mongoengine import StringField

db_name = 'article'
db_alias = 'article'
collection_name = 'article'
relation_fields = ['title', 'body']


class Article(Document):
    title = StringField(required=True)
    body = StringField(required=True)

    meta = {
        'db_alias' : db_alias,
        'collection' : collection_name,
        'strict' : False
    }

model_class = Article

