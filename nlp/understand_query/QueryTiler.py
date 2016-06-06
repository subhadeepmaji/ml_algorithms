from nlp.sense2vec import SenseEmbedding as SE


class QueryTiler:
    """
    Tile a query using a Sense Embedding model to compute the
    most likely labeling of the query sequence
    """
    def __init__(self, embedding):
        assert isinstance(embedding, SE.SenseEmbedding), \
            "embedding must be instance of SenseEmbedding class"
        self.embedding = embedding

    def __tile_single_token(self, token):
        words = self.embedding.get_tags_for_word(token)
        words = [w for w in words if w.split("|")[1] in self.embedding.senses]
        labels = [(w, self.embedding.model.most_similar(w, topn=1)[0][1]) for w in words]
        return max(labels, key=lambda e: e[1])[0]

    def tile(self, query, include_stopwords=False):
        query_tokens = query.split(" ")
        query_tokens = [self.embedding.stemmer.stem(w) for w in query_tokens]
        if not include_stopwords:
            query_tokens = [word for word in query_tokens if word not in self.embedding.stop_words]

        model_query_tokens = [w for w in query_tokens if self.embedding.get_tags_for_word(w)]
        non_model_query_tokens = list(set(query_tokens).difference(set(model_query_tokens)))

        tiling_sequence, tiling_cost = [None] * len(model_query_tokens), [-1] * len(model_query_tokens)
        tiling_cost[0] = 0
        tiling_sequence[0] = [self.__tile_single_token(model_query_tokens[0])]

        for index, word in enumerate(model_query_tokens[1:]):
            index += 1
            possible_labels = self.embedding.get_tags_for_word(word)
            max_likelihood, best_label, best_index = 0, None, None

            for label in possible_labels:
                current_tiling = " ".join(tiling_sequence[index - 1])
                current_tiling += (" " + label)
                tiling_likelihood = self.embedding.sequence_likelihood(current_tiling)

                max_likelihood = tiling_likelihood if tiling_likelihood > max_likelihood \
                    else max_likelihood
                best_label = label if max_likelihood == tiling_likelihood else best_label

            for j in xrange(index - 1, 0, -1):
                phrase = " ".join(model_query_tokens[j:index + 1])
                phrase_tags = self.embedding.get_tags_for_word(phrase)
                if not phrase_tags: continue
                for phrase_tag in phrase_tags:
                    if j - 1 > 0:
                        current_tiling = " ".join(tiling_sequence[j - 1])
                    current_tiling += (" " + phrase_tag)
                    tiling_likelihood = self.embedding.sequence_likelihood(current_tiling)

                    max_likelihood = tiling_likelihood if tiling_likelihood > max_likelihood \
                        else max_likelihood
                    best_label = phrase_tag if max_likelihood == tiling_likelihood else best_label
                    best_index = j - 1

            if best_index:
                tiling_sequence[index] = list(tiling_sequence[best_index])
            else:
                tiling_sequence[index] = list(tiling_sequence[index - 1])

            tiling_sequence[index].append(best_label)
            tiling_cost[index] = max_likelihood

        return tiling_sequence[-1], non_model_query_tokens











