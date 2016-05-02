from mongoengine import Document
from mongoengine import StringField

db_name = 'faq'
db_alias = 'faq'
collection_name = 'faq'
relation_fields = ['answer', 'question']


class FaqArticle(Document):
    question = StringField(required=True)
    answer = StringField(required=True)

    meta = {
        'db_alias' : db_alias,
        'collection' : collection_name,
        'strict' : False
    }

model_class = FaqArticle