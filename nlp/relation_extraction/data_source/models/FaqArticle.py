from mongoengine import Document
from mongoengine import StringField

db_name = 'faq'
db_alias = 'faq'
collection_name = 'faq'
relation_fields = ['question', 'answer']

field_weights = {
    'answer' : 2,
    'question': 1
}

function_fields  = None
payload_fields = None


class FaqArticle(Document):
    question = StringField(required=True)
    answer = StringField(required=True)

    meta = {
        'db_alias' : db_alias,
        'collection' : collection_name,
        'strict' : False
    }

model_class = FaqArticle