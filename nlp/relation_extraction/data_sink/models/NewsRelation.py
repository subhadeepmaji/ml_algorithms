from datetime import datetime
from elasticsearch_dsl import DocType, String, Date

index = 'relation'
mapping = 'news'


class Relation(DocType):

    leftEntity = String(analyzer='snowball')
    rightEntity = String(analyzer='snowball')
    relation = String(analyzer='snowball')
    text = String(index='not_analyzed')

    createdAt = Date()

    class Meta:
        index = index
        doc_type = mapping

    def save(self, **kwargs):
        if not self.createdAt:
            self.createdAt = datetime.utcnow()
        return super(Relation, self).save(**kwargs)

model_class = Relation

