import logging
import re
from itertools import chain
from nltk.corpus import stopwords
from practnlptools import tools

from nlp.embedding import DocumentEmbedding

logger = logging.getLogger(__name__)

SENT_RE = re.compile(r"([A-Z]*[^\.!?]*[\.!?])", re.M)
TAG_RE = re.compile(r"<[^>]+>")


class SentenceModelling(DocumentEmbedding.SentenceModel):

    def __init__(self, data_sources, workers, *args, **kwargs):
        """
        Sentence modelling
        :param data_sources: list of data sources to pull data from
        :param workers: number of processes to create in the pool
        """
        DocumentEmbedding.SentenceModel.__init__(self, *args, **kwargs)
        self.sources = data_sources
        self.annotator = tools.Annotator()
        self.workers = workers
        self.stop_words = set(stopwords.words('english'))

    def form_model(self):
        text_blocks = []
        for source in self.sources:
            source.start()

        logger.info("Reading the text blocks from the source")
        for item_tuple in chain(*self.sources):
            if not item_tuple:
                logger.warn("item read from source is empty")
                continue

            for item in item_tuple:
                if item == '': continue
                text_blocks.append(item)

        logger.info("Read all the text blocks")
        logger.info("Number of text blocks read : %d" % len(text_blocks))

        self.batch_train(text_blocks=text_blocks, tokenized=False)