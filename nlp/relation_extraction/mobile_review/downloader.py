#download GSMArena reviews and sync to MongoDB

from newspaper import Article
from mongoengine import connect
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, Future
from itertools import product
from collections import namedtuple

from spacy import English
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl.query import Q

from nlp.relation_extraction.data_source.models import ReviewArticle
from nlp.relation_extraction.data_sink.models import ReviewRelation
from nlp.relation_extraction.data_source import MONGO_HOST,MONGO_PORT
from nlp.relation_extraction.data_sink import ELASTIC_HOST, ELASTIC_PORT


url_base = "http://www.gsmarena.com/abc-review-%dp%d.php"
Review = namedtuple("Review", ["review_url", "review_title", "review_body"])
RelationQuery = namedtuple("RelationQuery", ["nouns", "verbs", "np_chunks"])


class GSMArenaDownloader:

    def __init__(self, readers=10, product_limit=2000, page_limit=10,
                 writers=5):
        self.readers = readers
        self.writers = writers
        self.product_limit = product_limit
        self.page_limit = page_limit
        self.mongo_connect = connect(ReviewArticle.db_name, ReviewArticle.db_alias,
                                     host=MONGO_HOST, port=MONGO_PORT)
        self.reader_pool = ThreadPoolExecutor(max_workers=self.readers)
        self.writer_pool = ThreadPoolExecutor(max_workers=self.writers)
        self.crawl_urls = []

    def form_urls(self):
        for prod_index, page_index in product(xrange(1, self.product_limit),
                                              xrange(1, self.page_limit)):
            page = url_base %(prod_index, page_index)
            self.crawl_urls.append(page)

    @staticmethod
    def __persist_review(review):
        review = ReviewArticle.ReviewArticle(url=review.review_url, title=review.review_title,
                                             text=review.review_body)
        try:
            review.save()
        except Exception as e:
            print "Error in saving review", e

    def __download_parse(self, page_url):
        article = Article(page_url)
        try:
            article.download()
            article.parse()
            article_url = article.url
            article_title = article.title
            if article.text == "":
                f = Future()
                f.done()
                return f

            meta = article.meta_data
            if meta.has_key("og") and meta["og"].has_key("url"):
                article_url = article.meta_data["og"]["url"]

            article_text = article.text.encode("ascii", errors='ignore').replace("\n", " ")
            review = Review(review_url=article_url, review_title=article_title, review_body=article_text)
            return self.writer_pool.submit(GSMArenaDownloader.__persist_review, review=review)

        except Exception as e:
            print "Error parsing the review page %s " %(page_url), e

    def download_reviews(self):
        print "Number of urls to crawl :: %d " %(len(self.crawl_urls))
        write_tasks = self.reader_pool.map(self.__download_parse, self.crawl_urls)
        wait(write_tasks, return_when=ALL_COMPLETED)



class QueryResolver:

    def __init__(self, num_results=3):
        self.num_results = num_results
        self.inited_engine = False

    def init_engine(self):
        # this initialization may take time, so moving this
        # out of object creation path
        self.nlp_engine = English()
        model_name = ReviewRelation.index + "." + ReviewRelation.mapping
        self.es_engine = connections.create_connection(model_name, hosts=[ELASTIC_HOST],
                                                       port=ELASTIC_PORT)
        self.inited_engine = True

    def form_relation_query(self, relation_query):
        search_engine = ReviewRelation.ReviewRelation.search(using=self.es_engine)

        entity_query = list(relation_query.nouns)
        entity_query.extend(relation_query.np_chunks)

        query = Q('bool', should=[Q('multi_match', query=entity_query,
                                    fields=['leftEntity', 'rightEntity']),
                                  Q('match', relation=relation_query.verbs)])

        s = search_engine.query(query)
        return s

    def resolve(self, query):
        assert self.inited_engine, "Query engine must be inited to perform a query"
        query = unicode(query)
        nlp_query = self.nlp_engine(query)

        noun_tokens, verb_tokens, noun_chunks = [], [], []
        for query_token in nlp_query:
            if query_token.pos_ == 'NOUN':
                noun_tokens.append(query_token)
            if query_token.pos_ == 'VERB':
                noun_tokens.append(query_token)

        for np_chunk in nlp_query.noun_chunks:
            noun_chunks.append(np_chunk)

        query = self.form_relation_query(RelationQuery(nouns=noun_tokens, verbs=verb_tokens,
                                                       np_chunks=noun_chunks))

        return query.execute()


