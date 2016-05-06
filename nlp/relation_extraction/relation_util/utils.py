
from nlp.relation_extraction import EntityTuple, PRONOUN_PHRASES, \
    POS_TAG_ENTITY_NP, POS_TAG_ENTITY_VP

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
porter_stemmer = PorterStemmer()


def normalize_entity(entity, chunk_parse, pos_tags, sense='NP', stem=True):
    """
    process a entity to form a normalized representation
    :param entity: entity to be normalized, tokenized into words
    :param chunk_parse: chunk parse of the entity
    :param pos_tags: pos tags of the tokens in the entity
    :param sense : sense of the parsing, can be noun phrases[NP] and verb phrases[VP],
    defaulting is noun phrases
    :param stem : stem the entity

    :return: normalized entity, return None if normalization leads
    to no entity been generated
    """

    intermediate_entities, normalized_entities = [], []
    current_entity = EntityTuple(None, [])
    if sense not in ['NP', 'VP']: raise RuntimeError("sense must be in [NP, VP]")

    pos_tags_sense = POS_TAG_ENTITY_NP if sense == 'NP' else POS_TAG_ENTITY_VP
    single, begin, end, intermediate = [e + sense for e in ['S-', 'B-', 'E-', 'I-']]
    stem_func = lambda word: porter_stemmer.stem(word) if stem else word

    for ((entity_index, entity_token), (_, chunk_tag)) in zip(enumerate(entity), chunk_parse):
        if chunk_tag == end:
            index = current_entity.index if current_entity.index is not None else entity_index
            current_entity.value.append(entity_token)
            intermediate_entities.append(EntityTuple(index, current_entity.value))
            current_entity = EntityTuple(None, "")

        elif chunk_tag == begin:
            current_entity = EntityTuple(entity_index, [entity_token])

        elif chunk_tag == single:
            intermediate_entities.append(EntityTuple(entity_index, [entity_token]))

        elif chunk_tag == intermediate:
            correct_index = current_entity.index if current_entity.index is not None \
                else entity_index
            current_entity.value.append(entity_token)
            current_entity = EntityTuple(correct_index, current_entity.value)

        elif chunk_tag in PRONOUN_PHRASES:
            if current_entity.index:
                current_entity.value.append(entity_token)
                current_entity = EntityTuple(current_entity.index, current_entity.value)

    if current_entity not in intermediate_entities:
        intermediate_entities.append(current_entity)

    for entity_tuple in intermediate_entities:
        entity_text = ""
        for index, entity_token in enumerate(entity_tuple.value):
            if pos_tags[entity_tuple.index + index][1] in pos_tags_sense:
                entity_text += (" " + stem_func(entity_token))
        if entity_text: normalized_entities.append(entity_text.strip())

    if not normalized_entities:
        return None

    return " ".join(normalized_entities)


def normalize_relation(relation_phrase):
    """
    normalize a relation verb
    :param relation_phrase: relation verb phrase to be normalized
    :return: normalized form of the verb
    """
    return " ".join([porter_stemmer.stem(w) for w in word_tokenize(relation_phrase)])


def sublist_find(haystack, needle):
    """
    Find a needle in a haystack, aka sublist find in a bigger list
    :param haystack: bigger list
    :param needle: list to find
    :return: return the start index of the smaller list in bigger list,
    -1 if not found
    """
    matched, start_index = 0, -1
    for (hay_index, hay_ele) in enumerate(haystack):
        if hay_ele != needle[matched]:
            matched = 0
            start_index = -1
        else:
            matched += 1
            if start_index == -1: start_index = hay_index
            if matched == len(needle): return start_index

    return start_index


def word_tokenize_entity(words, entity):
    """
    Tokenize an entity string using the words in the given word list,
    uses DP to compute optimal complete split of the entity string
    :param words: list of words to tokenize the string by
    :param entity: the string to be tokenized
    :return: return the tokenized entity as list of tokens
    """
    # append white space character to the words list
    words.append(" ")
    entity_char_tokens = [c for c in entity]
    tokenized_entity, split_index = [], []
    cover = [False] * (len(entity_char_tokens))

    for index, entity_c in enumerate(entity_char_tokens):
        assignments = [(j, cover[j] and words.count("".join(entity_char_tokens[j + 1: index + 1])))
                       for j in xrange(index - 1, -1, -1)]

        assignments.append((-1, words.count("".join(entity_char_tokens[0: index + 1]))))
        if not assignments:
            cover[index] = False
        else:
            token_segment = max(assignments, key=lambda e: e[1])
            cover[index] = token_segment[1]
            if cover[index]: split_index.append((index, token_segment[0]))

    end, start = split_index[-1]
    split_index = dict(split_index)
    while True:
        word_found = ''.join(entity_char_tokens[start + 1: end + 1])
        if word_found not in [' ']: tokenized_entity.append(word_found)
        end, start = start, split_index.get(start)
        if not start: break

    words.remove(" ")
    return tokenized_entity[::-1]


def form_entity(sentence_as_tokens, arg_text, chunk_parse, pos_tags,
                sense='NP', stem=True):
    """
    Form a entity from a argument string using the parsed chunks and pos tags
    and the sentence tokenization of the sentence from which the argument is generated
    :param sentence_as_tokens: tokenized sentence generating the argument
    :param arg_text: argument string
    :param chunk_parse: chunking of the argument
    :param pos_tags: pos of the argument
    :param sense: sense of parsing, NP/VP
    :param stem: whether to stem the entity

    :return: string of the normalized entity
    """
    tokenized_arg = word_tokenize_entity(sentence_as_tokens, arg_text)
    arg_index = sublist_find(sentence_as_tokens, tokenized_arg)

    arg_chunk_parse = chunk_parse[arg_index: arg_index + len(tokenized_arg)]
    arg_pos_tag = pos_tags[arg_index: arg_index + len(tokenized_arg)]

    return normalize_entity(tokenized_arg, arg_chunk_parse, arg_pos_tag, sense=sense, stem=stem)
