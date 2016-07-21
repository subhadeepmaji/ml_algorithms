from datetime import datetime
from elasticsearch_dsl import DocType, String, Date

index = 'relation'
mapping = 'reviews'


class ReviewRelation(DocType):

    leftEntity = String(analyzer='snowball')
    rightEntity = String(analyzer='snowball')
    relation = String(analyzer='snowball')
    webLocation = String(index='not_analyzed')
    productName = String(analyzer='snowball')
    text = String(index='not_analyzed')

    createdAt = Date()

    class Meta:
        index = index
        doc_type = mapping

    def save(self, **kwargs):
        if not self.createdAt:
            self.createdAt = datetime.utcnow()
        return super(ReviewRelation, self).save(**kwargs)

model_class = ReviewRelation
