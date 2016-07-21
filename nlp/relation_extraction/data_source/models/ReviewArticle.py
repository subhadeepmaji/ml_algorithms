from mongoengine import Document
from mongoengine import StringField

db_name = 'reviews'
db_alias = 'reviews'
collection_name = 'phone_reviews'
relation_fields = ['text']
function_fields  = ['title']
payload_fields = ['url']


class ReviewArticle(Document):
    url = StringField(required=True)
    title = StringField(required=True)
    text = StringField(required=True)
    productName = StringField(required=True)

    meta = {
        'db_alias' : db_alias,
        'collection' : collection_name,
        'strict' : False
    }

model_class = ReviewArticle
