#download GSMArena reviews and sync to MongoDB

from newspaper import Article
from mongoengine import connect
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, Future
from itertools import product
from collections import namedtuple
from itertools import chain

from spacy import English
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl.query import Q, Match

from collections import defaultdict
from nlp.relation_extraction.data_source.models import ReviewArticle
from nlp.relation_extraction.data_sink.models import ReviewRelation
from nlp.relation_extraction.data_source import MONGO_HOST,MONGO_PORT
from nlp.relation_extraction.data_sink import ELASTIC_HOST, ELASTIC_PORT


url_base = "http://www.gsmarena.com/abc-review-%dp%d.php"
Review = namedtuple("Review", ["review_url", "review_title", "review_body"])
RelationQuery = namedtuple("RelationQuery", ["nouns", "verbs", "np_chunks",
                                             "product", "attributes"])


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
        for prod_index, page_index in product(xrange(self.product_limit, 1, -1),
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


class EntityNormalizer:

    def __init__(self, kb_triples, sim_threshold=0.6):
        self.kb_triples = kb_triples
        self.entities = defaultdict(list)
        self.entities_text = set()
        self.relations = defaultdict(list)
        self.relation_text = set()
        self.nlp_engine = English()
        self.stemmer = PorterStemmer()
        self.concept_similarity = defaultdict(list)
        self.sim_threshold = sim_threshold

    def normalize_entities(self):
        for triple in self.kb_triples:

            left_entity, rel, right_entity, sentence, product = \
                triple.left_entity, triple.relation, triple.right_entity, \
                triple.sentence, triple.payload

            stem_tokens = {self.stemmer.stem(word) : word for word in word_tokenize(sentence)}

            left_entity  = " ".join([stem_tokens[w] for w in word_tokenize(left_entity)
                                     if w in stem_tokens])
            right_entity = " ".join([stem_tokens[w] for w in word_tokenize(right_entity)
                                     if w in stem_tokens])
            rel = " ".join([stem_tokens[w] for w in word_tokenize(rel) if w in stem_tokens])

            if not left_entity or not right_entity or not rel:
                continue

            if len(left_entity.split(" ")) < 5:
                if (product, left_entity) not in self.entities_text:
                    left_entity = self.nlp_engine(unicode(left_entity))
                    self.entities[product].append(left_entity)
                    self.entities_text.add((product, left_entity.text))

            if len(right_entity.split(" ")) < 5:
                if (product, right_entity) not in self.entities_text:
                    right_entity = self.nlp_engine(unicode(right_entity))
                    self.entities[product].append(right_entity)
                    self.entities_text.add((product, right_entity.text))

            if (product, rel) not in self.relation_text:
                rel = self.nlp_engine(unicode(rel))
                self.relations[product].append(rel)
                self.relation_text.add((product, rel.text))

    def form_similar_concepts(self, concepts, topn=10):
        entities = chain(*self.entities.values())
        for concept in concepts:
            concept = self.nlp_engine(unicode(concept))
            concept_similarity = [(entity, concept.similarity(entity)) for entity in entities]
            concept_similarity = [(entity.text, sim) for entity, sim in concept_similarity
                                  if sim > self.sim_threshold]
            concept_similarity = sorted(concept_similarity, key=lambda e: -e[1])
            concept_similarity = [c[0] for c in concept_similarity][:topn]
            self.concept_similarity[concept.text] = concept_similarity


    def most_similar(self, query, topn=10):
        nouns = list(query.nouns)
        nouns.append(query.np_chunks)
        verbs = "" if not query.verbs else query.verbs

        if not nouns: nouns = ""
        if not verbs and not nouns: return None

        sim_nouns = []
        sim_verbs = []

        entities  = self.entities[query.product]
        relations = self.entities[query.product]

        for noun in nouns:
            noun = self.nlp_engine(unicode(noun))
            noun_similarity = [(entity, noun.similarity(entity)) for entity in entities]
            noun_similarity = [(entity.text, sim) for entity, sim in noun_similarity
                               if sim > self.sim_threshold]
            sim_nouns.extend(noun_similarity)

        for verb in verbs:
            verb = self.nlp_engine(unicode(verb))
            verb_similarity = [(relation, verb.similarity(relation)) for relation in relations]
            verb_similarity = [(relation.text, sim) for relation, sim in verb_similarity
                               if sim > self.sim_threshold]
            sim_verbs.extend(verb_similarity)

        noun_similarity = sorted(sim_nouns, key=lambda e: -e[1])
        verb_similarity = sorted(sim_verbs, key=lambda e: -e[1])

        noun_similarity = [n[0] for n in noun_similarity]
        verb_similarity = [v[0] for v in verb_similarity]

        entity_closeness = []
        if query.attributes:
            for attribute in query.attributes:
                close_entities = self.concept_similarity[attribute]
                entity_closeness.extend(close_entities)

        return  noun_similarity[:topn], verb_similarity[:topn], entity_closeness


class QueryResolver:

    def __init__(self, kb_triples, entity_normalizer, num_results=5):
        self.num_results = num_results
        self.kb_triples = kb_triples
        self.entity_normalizer = entity_normalizer
        self.nlp_engine = entity_normalizer.nlp_engine
        self.inited_engine = False

    def init_engine(self):
        # this initialization may take time, so moving this
        # out of object creation path
        model_name = ReviewRelation.index + "." + ReviewRelation.mapping
        self.es_engine = connections.create_connection(model_name, hosts=[ELASTIC_HOST],
                                                       port=ELASTIC_PORT)
        self.inited_engine = True

    def form_relation_query(self, relation_query):
        search_engine = ReviewRelation.ReviewRelation.search(using=self.es_engine)

        entity_query = list(relation_query.nouns)
        entity_query.extend(relation_query.np_chunks)
        entity_query.extend(relation_query.attributes)

        if not entity_query: entity_query = ""
        verb_query = "" if not relation_query.verbs else relation_query.verbs

        if not entity_query and not verb_query: return None

        product = relation_query.product
        query = Q('bool',
                  must=[Q('match', productName={"query" : product, "type" : "phrase"})],
                  should=[Q('multi_match', query=entity_query,
                            fields=['leftEntity', 'rightEntity'], boost=3),
                          Match(relation={"query" : verb_query})])

        s = search_engine.query(query)
        return s

    def resolve(self, query, attributes, product):
        assert self.inited_engine, "Query engine must be inited to perform a query"
        query = unicode(query)
        nlp_query = self.nlp_engine(query)

        noun_tokens, verb_tokens, noun_chunks = [], [], []
        for query_token in nlp_query:
            print query_token, query_token.pos_
            if query_token.pos_ == 'NOUN':
                noun_tokens.append(query_token.text)
            if query_token.pos_ == 'VERB':
                verb_tokens.append(query_token.text)

        for np_chunk in nlp_query.noun_chunks:
            noun_chunks.append(np_chunk.text)

        query = RelationQuery(nouns=noun_tokens, verbs=verb_tokens,
                              np_chunks=noun_chunks, product=product,
                              attributes=attributes)

        np, vp, close_concepts = self.entity_normalizer.most_similar(query)
        query = RelationQuery(nouns = np, verbs = vp, np_chunks=[],
                              product=product, attributes=close_concepts)

        query = self.form_relation_query(query)
        if not query: return None

        response = query.execute()
        query_response = []

        for doc in response[:self.num_results]:
            print "----------------------------------"

            print "product name :: ", doc.productName
            print "entity :: ", doc.leftEntity, doc.rightEntity
            print "relation :: ", doc.relation
            print "sentence :: ", doc.sentence

            print "----------------------------------"
            query_response.append({
                "productName" : doc.productName,
                "sentence" : doc.sentence,
                "reviewText" : doc.text
            })

        return query_response



