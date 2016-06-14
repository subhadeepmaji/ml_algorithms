import logging, re
from itertools import chain, izip
from util import LoggerConfig
from enum import Enum
from nltk import word_tokenize as tokenizer
from nlp.relation_extraction.relation_util import utils as relation_util


config = dict(LoggerConfig.logger_config)
config['filename'] = 'sense2vec.log'

CONJUNCTION = {'CC', 'IN'}
CARDINAL = {'CD'}
ADJECTIVE = {'JJ', 'JJR', 'JJS'}
NOUN = {'NN', 'NNS', 'NNP', 'NNPS'}
VERB = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
SYMBOL = {'SYM'}

SENT_RE = re.compile(r"([A-Z]*[^\.!?]*[\.!?])", re.M)
TAG_RE = re.compile(r"<[^>]+>")

logging.basicConfig(**config)
logger = logging.getLogger(__name__)


class VPTags(Enum):
    single = 'S-VP'
    begin = 'B-VP'
    intermediate = 'I-VP'
    end = 'E-VP'


class NPTags(Enum):
    single = 'S-NP'
    begin = 'B-NP'
    intermediate = 'I-NP'
    end = 'E-NP'

phrase_tags = set([e.value for e in chain(*[NPTags, VPTags])])
alpha_numeric = re.compile('^\w+$')


def form_phrases(chunk_parse, sense):

    if sense not in [NPTags, VPTags]:
        raise RuntimeError("Sense must be NPTags or VPTags Enum")

    current_sense, phrases = [], []
    for word, chunk_tag in chunk_parse:
        if chunk_tag == sense.single.value:
            phrases.append(word)

        if chunk_tag in [sense.begin.value, sense.intermediate.value]:
            current_sense.append(word)

        if chunk_tag == sense.end.value:
            current_sense.append(word)
            phrases.append(" ".join(current_sense))
            current_sense = []

    return phrases


def normalize_pos(pos_tag):

    if pos_tag in CONJUNCTION:
        return 'CONJUNCTION'
    elif pos_tag in CARDINAL:
        return 'CARDINAL'
    elif pos_tag in ADJECTIVE:
        return 'ADJECTIVE'
    elif pos_tag in NOUN:
        return 'NOUN'
    elif pos_tag in VERB:
        return 'VERB'
    else:
        return 'DEFAULT'


def word_tokenize(text_block, stemmer, stop_words):
    sentences = SENT_RE.findall(text_block)
    sense_phrases = []
    for sentence in sentences:
        sentence = sentence.replace('\'', '').replace('(', ' ') \
            .replace(')', ' ').replace("/", " or ").replace("-", "")

        sentence = TAG_RE.sub('', sentence)
        sentence = "".join((c for c in sentence if 0 < ord(c) < 127))
        sentence_words = [stemmer.stem(word) for word in tokenizer(sentence) if word not in stop_words
                          and re.match(alpha_numeric, word)]
        sense_phrases.append(sentence_words)
        logger.info("Will sense tokenize : %s" % sentence)
    return sense_phrases


def sense_tokenize(text_block, annotator, stemmer, stop_words):
    """
    tokenize a block into sentences which are word tokenized, preserving the sense of the words
    (see the original paper for details)
    :param text_block: block of text (string)
    :param annotator: senna annotator
    :param stemmer: porter stemmer instance
    :param stop_words: list of stopwords to use
    :param phrase_tags: tags of the phrases to parse
    :return: list of sentences each tokenized into words
    """
    sentences = SENT_RE.findall(text_block)
    sense_phrases = []

    for sentence in sentences:
        sentence = sentence.replace('\'', '').replace('(', ' ') \
            .replace(')', ' ').replace("/", " or ").replace("-", "")

        sentence = TAG_RE.sub('', sentence)
        sentence = "".join((c for c in sentence if 0 < ord(c) < 127))
        #logger.info("Will sense tokenize : %s" % sentence)
        try:
            senna_annotation = annotator.getAnnotations(sentence)
        except Exception as e:
            #logger.error("annontator error")
            #logger.error(e)
            continue

        chunk_parse, pos_tags, words = senna_annotation['chunk'], senna_annotation['pos'], \
                                       senna_annotation['words']

        single_words = [stemmer.stem(word) + '|' + normalize_pos(tag)
                        for word, tag in pos_tags if word not in stop_words]

        sense_phrases.append(single_words)

        noun_phrases = form_phrases(chunk_parse, NPTags)
        verb_phrases = form_phrases(chunk_parse, VPTags)

        non_phrase_words = [stemmer.stem(word) + '|' + normalize_pos(pos_tag) for
                            ((word, chunk_tag), (_, pos_tag)) in izip(chunk_parse, pos_tags)
                            if chunk_tag not in phrase_tags if word not in stop_words]

        noun_entities, verb_entities = [], []
        for np in noun_phrases:
            en = relation_util.form_entity(words, np, chunk_parse, pos_tags, 'NP')
            if not en: continue
            en_words = en.split(" ")
            if len(en_words) > 1:
                en_words_with_pos = [stemmer.stem(w) + '|' + normalize_pos(pos_tag)
                                     for (w, pos_tag) in pos_tags if stemmer.stem(w) in en_words]
                en_words_with_pos.append(en + '|NP')
                sense_phrases.append(en_words_with_pos)
            noun_entities.append(en + '|NP')

        for vp in verb_phrases:
            en = relation_util.form_entity(words, vp, chunk_parse, pos_tags, 'VP')
            if not en: continue
            en_words = en.split(" ")
            if len(en_words) > 1:
                en_words_with_pos = [stemmer.stem(w) + '|' + normalize_pos(pos_tag)
                                     for (w, pos_tag) in pos_tags if stemmer.stem(w) in en_words]
                en_words_with_pos.append(en + '|VP')
                sense_phrases.append(en_words_with_pos)
            verb_entities.append(en + '|VP')

        noun_index, verb_index, non_phrase_index = 0, 0, 0
        sense_words = []
        for (word, chunk_tag) in chunk_parse:
            if chunk_tag not in phrase_tags:
                if non_phrase_index < len(non_phrase_words):
                    sense_words.append(non_phrase_words[non_phrase_index])
                    non_phrase_index += 1

            if chunk_tag in [NPTags.end.value, NPTags.single.value]:
                if noun_index < len(noun_entities):
                    sense_words.append(noun_entities[noun_index])
                    noun_index += 1

            if chunk_tag in [VPTags.end.value, VPTags.single.value]:
                if verb_index < len(verb_entities):
                    sense_words.append(verb_entities[verb_index])
                    verb_index += 1

        if sense_words:
            sense_phrases.append(sense_words)

    return sense_phrases


