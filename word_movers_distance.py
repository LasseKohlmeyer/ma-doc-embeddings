import json
import os
from typing import Union

from gensim import corpora
from gensim.models import KeyedVectors, TfidfModel
from gensim.similarities import WmdSimilarity

from corpus_iterators import CorpusDocumentIterator
from corpus_structure import Corpus


class WordMoversDistance:
    embedding_path = 'E:/embeddings/glove.6B.300d.txt'

    def __init__(self, corpus: Corpus, embedding_path: str, top_n_docs: int = 10,
                 top_n_words: int = 1000):
        self.corpus = corpus
        self.embedding_path = embedding_path
        self.top_n_docs = top_n_docs
        self.top_n_words = top_n_words
        self.similarities = None
        self.wmd_corpus = None
        self.doc_id_mapping = None
        self.reverse_doc_id_mapping = None
        self.word_movers_distance_topn()

    def word_movers_distance_topn(self):
        model = KeyedVectors.load_word2vec_format(self.embedding_path, binary=False)
        tokenized_document_corpus = CorpusDocumentIterator(self.corpus, lemma=False, lower=False)

        dictionary = corpora.Dictionary()
        bow_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in tokenized_document_corpus]
        tf_idf_model = TfidfModel(bow_corpus)

        wmd_corpus = []
        doc_id_mapping = {doc_id: i for i, doc_id in enumerate(tokenized_document_corpus.doc_ids)}
        for doc in tf_idf_model[bow_corpus]:
            tuples = [(dictionary[word_id], sim) for word_id, sim in doc]
            tuples.sort(key=lambda x: x[1])
            tuples = tuples[:self.top_n_words]
            relevant_words = [word for word, sim in tuples]
            wmd_corpus.append(relevant_words)

        similarities = WmdSimilarity(wmd_corpus, model.wv, num_best=self.top_n_docs)
        # print(similarities[wmd_corpus[0]])
        # print(similarities[wmd_corpus[doc_id_mapping["cb_0"]]])
        self.similarities = similarities
        self.wmd_corpus = wmd_corpus
        self.doc_id_mapping = doc_id_mapping
        self.reverse_doc_id_mapping = tokenized_document_corpus.doc_ids

    def most_similar(self, doc_id: Union[str, int]):
        if isinstance(doc_id, str):
            sims = self.similarities[self.wmd_corpus[self.doc_id_mapping[doc_id]]]
            sims = [(self.reverse_doc_id_mapping[doc_int_id], sim) for doc_int_id, sim in sims]
        else:
            sims = self.similarities[self.wmd_corpus[doc_id]]
        return sims

    def store_similarities(self, path: str):
        sims_to_store = {}
        for doc_id in self.corpus.documents.keys():
            sims_to_store[doc_id] = self.most_similar(doc_id)

        with open(path, 'w', encoding="utf-8") as fp:
            json.dump(sims_to_store, fp, indent=1)
        return sims_to_store

    @staticmethod
    def load_similarities(path: str):
        with open(path, 'r', encoding="utf-8") as fp:
            sims = json.load(fp)

        sims = {doc_id: [(tup[0], tup[1]) for tup in sim] for doc_id, sim in sims.items()}

        return sims

    @classmethod
    def similarities(cls, path: str = None, corpus: Corpus = None, top_n_docs: int = None, top_n_words: int = None):
        if os.path.isfile(path):
            return WordMoversDistance.load_similarities(path)
        else:
            wmd_obj = WordMoversDistance(corpus=corpus,
                                         embedding_path=cls.embedding_path,
                                         top_n_docs=top_n_docs,
                                         top_n_words=top_n_words)
            return wmd_obj.store_similarities(path)


if __name__ == "__main__":
    # data_set_name = "classic_gutenberg"
    # c = Corpus.fast_load(path=os.path.join('corpora', data_set_name), load_entities=False)
    # # wmd = WordMoversDistance(corpus=c, embedding_path='E:/embeddings/glove.6B.100d.txt', top_n_docs=10, top_n_words=10)
    # # print(wmd.most_similar("cb_0"))
    #
    # sims = WordMoversDistance.similarities(path="D:/models/wmd/test1.json",
    #                                        corpus=c,
    #                                        top_n_docs=10,
    #                                        top_n_words=10)
    # print(sims["cb_0"])
    import numpy as np
    from sklearn.cross_decomposition import CCA

    X = [[0., 0., 1., 2.], [1., 0., 0., 3.], [2., 2., 2., 4.], [3., 5., 4., 5.]]
    Y = [[0.1, -0.2, 3.], [0.9, 1.1, 3.], [6.2, 5.9, 3.], [11.9, 12.3, 3.]]
    Z = [[0.1, -0.2, 3.], [0.9, 1.1, 3.], [6.2, 5.9, 3.], [11.9, 12.3, 3.]]
    cca = CCA(n_components=3)
    cca.fit(X, Y)

    X_c, Y_c = cca.transform(X, Y)
    print(X_c)
    print(Y_c)