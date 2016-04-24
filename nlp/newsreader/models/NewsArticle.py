import hashlib, datetime
from mongoengine import Document
from mongoengine import StringField, ListField, DateTimeField, \
    DictField, URLField

class NewsArticle(Document):
    source     = StringField(required = True)
    title      = StringField(required = True)
    location   = URLField(required = True)
    text       = StringField(required = True)
    
    summary      = StringField()
    keywords     = ListField(field = StringField(max_length = 2000))
    mod_time     = DateTimeField(default = datetime.datetime.now)
    publish_date = DateTimeField()
    uniq_id    = StringField(unique = True, required = True)
    open_graph = DictField()
    meta       = {'db_alias' : 'news',
                  "index_background" : True,
                  "indexes" : [
                      "#uniq_id", # hash index on uniq_id
                      "$title"    # text index on title 
                  ]}

    def clean(self):
        self.uniq_id = hashlib.sha256(self.location).hexdigest()
        


