import os
from collections import defaultdict
from typing import Union, List

from gensim.models import Doc2Vec
from tqdm import tqdm

from corpus_structure import Corpus, Document
from doc2vec_structures import DocumentKeyedVectors
from topic_modelling import TopicModeller
import pandas as pd
import time
from scipy import stats
import numpy as np
from vectorization_utils import Vectorization


class FacetEfficientEvaluation:
    def __init__(self, vectors: Union[Doc2Vec, DocumentKeyedVectors], corpus: Corpus,
                 data_set_name: str, facet_names: List[str] = None, topic_vectors=None):
        self.vectors = vectors
        self.corpus = corpus
        # self.same_facet_words = same_facet_words
        self.lemma = False
        self.lower = False
        self.data_set_name = data_set_name
        self.topic_vectors = topic_vectors
        if facet_names is None:
            self.facets = [
                "loc",
                "time",
                "atm",
                "sty",
                "cont"
            ]
        else:
            self.facets = facet_names

    def get_facet_words(self, document: Document, facet_name: str):
        facet_words = []
        if facet_name.lower() == "loc":
            try:
                facet_words.extend(document.doc_entities["LOC"])
            except KeyError:
                pass
            try:
                facet_words.extend(document.doc_entities["FAC"])
            except KeyError:
                pass
            try:
                facet_words.extend(document.doc_entities["GPE"])
            except KeyError:
                pass
            # print([word for word in facet_words])
            facet_words = [word[2].representation() for word in facet_words]
        elif facet_name.lower() == "time":
            try:
                facet_words.extend(document.doc_entities["DATE"])
            except KeyError:
                pass
            try:
                facet_words.extend(document.doc_entities["TIME"])
            except KeyError:
                pass
            try:
                facet_words.extend(document.doc_entities["EVENT"])
            except KeyError:
                pass
            facet_words = [word[2].representation() for word in facet_words]
        elif facet_name.lower() == "atm":
            facet_words.extend(document.get_flat_and_filtered_document_tokens(lemma=self.lemma,
                                                                              lower=self.lower,
                                                                              pos=["ADJ", "ADV"]))
        elif facet_name.lower() == "sty":
            facet_words.extend(document.get_flat_and_filtered_document_tokens(lemma=self.lemma,
                                                                              lower=self.lower,
                                                                              focus_stopwords=True))
        else:
            raise UserWarning(f"Not supported facet name >{facet_name}<!")
        return facet_words

    def topic_evaluation(self, facet_name: str, doc_id: str, is_series_corpus: bool):
        sim_docs = Vectorization.most_similar_documents(self.vectors, self.corpus, positives=doc_id,
                                                        topn=len(self.corpus.documents.items()),
                                                        feature_to_use=facet_name, print_results=False,
                                                        series=is_series_corpus)

        sim_docs = {doctag.replace(f'_{facet_name}', ''): sim for (doctag, sim) in sim_docs}
        topic_sim_docs = Vectorization.most_similar_documents(self.topic_vectors, self.corpus, positives=doc_id,
                                                              topn=len(self.corpus.documents.items()),
                                                              print_results=False,
                                                              series=is_series_corpus)
        topic_sim_docs = {doctag: sim for (doctag, sim) in topic_sim_docs}

        neural_sims = []
        topic_sims = []
        for doc_key in self.corpus.documents.keys():
            neural_sims.append(sim_docs[doc_key])
            topic_sims.append(topic_sim_docs[doc_key])

        spearman_corr, spearman_p = stats.spearmanr(np.array(neural_sims), np.array(topic_sims))
        spearman_corr = abs(spearman_corr)
        spearman_corr_strict = spearman_corr
        if spearman_p >= 0.05:
            spearman_corr_strict = 0

        return spearman_corr, spearman_corr_strict

    def word_neighborhood_evaluation(self, facet_name: str, doc_id: str, doc_top_n: int, word_top_n: int,
                                     is_series_corpus: bool, document: Document):
        sim_docs = Vectorization.most_similar_documents(self.vectors, self.corpus, positives=doc_id,
                                                        topn=doc_top_n,
                                                        feature_to_use=facet_name, print_results=False,
                                                        series=is_series_corpus)

        # print(doc_id, len(sim_docs))
        sim_words = Vectorization.most_similar_words(self.vectors, positives=[doc_id],
                                                     topn=word_top_n,
                                                     feature_to_use=facet_name, print_results=False)
        sim_words_relaxed = set([word for word, sim in sim_words])
        # print(sim_words, len(sim_words))
        facet_words = self.get_facet_words(document, facet_name)
        sim_words_strict = set([word for word in sim_words_relaxed
                                if word in facet_words])
        # print(sim_words)
        # print(facet_doc_id)
        # print(sim_docs, sim_words)
        # print('_____')
        shared_word_values_relaxed = []
        shared_word_values_strict = []
        reciprocal_ranks = []
        r = 1

        for sim_doc_id, sim_doc in sim_docs:
            if str(sim_doc_id).startswith(doc_id):
                continue
            # print('>>', sim_doc_id, word_top_n)
            sim_doc_words = Vectorization.most_similar_words(self.vectors, positives=[sim_doc_id],
                                                             topn=word_top_n,
                                                             feature_to_use=None, print_results=False)
            # print(len(sim_doc_words))
            # print(sim_words_relaxed)
            # print(sim_doc_words)
            sim_doc_words_relaxed = set([word for word, sim in sim_doc_words])
            sim_doc_words_strict = set([word for word in sim_doc_words_relaxed if word in facet_words])
            # print()
            shared_words_relaxed = sim_words_relaxed.intersection(sim_doc_words_relaxed)
            shared_words_strict = sim_words_strict.intersection(sim_doc_words_strict)
            # print(len(sim_words_strict), len(sim_doc_words), len(shared_words_relaxed),
            #       len(shared_words_strict))
            # print(sim_words_relaxed)
            # print(sim_doc_words_relaxed)
            #
            # print('sty', facet_words)
            # print(sim_words_strict)
            # print(sim_doc_words_strict)
            # print()
            shared_word_values_relaxed.append(len(shared_words_relaxed) / word_top_n)
            shared_word_values_strict.append(len(shared_words_strict) / word_top_n)
            reciprocal_ranks.append(1 / r)
            r += 1
            # print(len(shared_words), shared_words)

        reciprocal_ranks = [rank / sum(reciprocal_ranks) for rank in reciprocal_ranks]
        score_relaxed = sum([shared_val * rank_val
                             for shared_val, rank_val in zip(shared_word_values_relaxed, reciprocal_ranks)])
        score_strict = sum([shared_val * rank_val
                            for shared_val, rank_val in zip(shared_word_values_strict, reciprocal_ranks)])

        return score_relaxed, score_strict

    def evaluate(self, doc_top_n: int = 20, word_top_n: int = 1000):
        facet_eval_scores_relaxed = defaultdict(list)
        facet_eval_scores_strict = defaultdict(list)

        for doc_id, document in tqdm(self.corpus.documents.items(),
                                     desc="Iterate over documents.",
                                     total=len(self.corpus.documents)):
            # facet_doc_id = f'{doc_id}_{self.facet_name}'
            # print(doc_id)
            document.load_sentences_from_disk()
            document.set_entities()
            # print(document.doc_entities.keys())

            if "series" in self.data_set_name:
                is_series_corpus = True
            else:
                is_series_corpus = False

            for facet_name in self.facets:
                if facet_name == "cont":
                    score_relaxed, score_strict = self.topic_evaluation(facet_name, doc_id, is_series_corpus)
                else:
                    score_relaxed, score_strict = self.word_neighborhood_evaluation(facet_name, doc_id, doc_top_n,
                                                                                    word_top_n,
                                                                                    is_series_corpus, document)
                    # print(score)
                facet_eval_scores_relaxed[facet_name].append(score_relaxed)
                facet_eval_scores_strict[facet_name].append(score_strict)

            document.sentences = None
            document.doc_entities = None
        return facet_eval_scores_relaxed, facet_eval_scores_strict


# class FacetEvaluation:
#     def __init__(self, facet_name: str, vectors: Union[Doc2Vec, DocumentKeyedVectors], corpus: Corpus,
#                  data_set_name: str):
#         self.facet_name = facet_name
#         self.vectors = vectors
#         self.corpus = corpus
#         # self.same_facet_words = same_facet_words
#         self.lemma = False
#         self.lower = False
#         self.data_set_name = data_set_name
#
#     def get_facet_words(self, document: Document):
#         facet_words = []
#         if self.facet_name.lower() == "loc":
#             try:
#                 facet_words.extend(document.doc_entities["LOC"])
#             except KeyError:
#                 pass
#             try:
#                 facet_words.extend(document.doc_entities["FAC"])
#             except KeyError:
#                 pass
#             try:
#                 facet_words.extend(document.doc_entities["GPE"])
#             except KeyError:
#                 pass
#             # print([word for word in facet_words])
#             facet_words = [word[2].representation() for word in facet_words]
#         elif self.facet_name.lower() == "time":
#             try:
#                 facet_words.extend(document.doc_entities["DATE"])
#             except KeyError:
#                 pass
#             try:
#                 facet_words.extend(document.doc_entities["TIME"])
#             except KeyError:
#                 pass
#             try:
#                 facet_words.extend(document.doc_entities["EVENT"])
#             except KeyError:
#                 pass
#             facet_words = [word[2].representation() for word in facet_words]
#         elif self.facet_name.lower() == "atm":
#             facet_words.extend(document.get_flat_and_filtered_document_tokens(lemma=self.lemma,
#                                                                               lower=self.lower,
#                                                                               pos=["ADJ", "ADV"]))
#         elif self.facet_name.lower() == "sty":
#             facet_words.extend(document.get_flat_and_filtered_document_tokens(lemma=self.lemma,
#                                                                               lower=self.lower,
#                                                                               focus_stopwords=True))
#         else:
#             raise UserWarning(f"Not supported facet name >{self.facet_name}<!")
#         return facet_words
#
#     def evaluate(self, doc_top_n: int = 20, word_top_n: int = 100):
#         eval_scores_relaxed = []
#         eval_scores_strict = []
#
#         for doc_id, document in tqdm(self.corpus.documents.items(),
#                                      desc="Iterate over documents.",
#                                      total=len(self.corpus.documents)):
#             # facet_doc_id = f'{doc_id}_{self.facet_name}'
#             # print(doc_id)
#             document.load_sentences_from_disk()
#             document.set_entities()
#             # print(document.doc_entities.keys())
#
#             if "series" in self.data_set_name:
#                 is_series_corpus = True
#             else:
#                 is_series_corpus = False
#             sim_docs = Vectorizer.most_similar_documents(self.vectors, self.corpus, positives=doc_id,
#                                                          topn=doc_top_n,
#                                                          feature_to_use=self.facet_name, print_results=False,
#                                                          series=is_series_corpus)
#             # print(doc_id, len(sim_docs))
#             sim_words = Vectorizer.most_similar_words(self.vectors, positives=[doc_id],
#                                                       topn=word_top_n,
#                                                       feature_to_use=self.facet_name, print_results=False)
#             sim_words_relaxed = set([word for word, sim in sim_words])
#             facet_words = self.get_facet_words(document)
#             sim_words_strict = set([word for word in sim_words_relaxed
#                                     if word in facet_words])
#             # print(sim_words)
#             # print(facet_doc_id)
#             # print(sim_docs, sim_words)
#             # print('_____')
#             shared_word_values_relaxed = []
#             shared_word_values_strict = []
#             reciprocal_ranks = []
#             r = 1
#
#             for sim_doc_id, sim_doc in sim_docs:
#                 if str(sim_doc_id).startswith(doc_id):
#                     continue
#                 print('>>', sim_doc_id, word_top_n)
#                 sim_doc_words = Vectorizer.most_similar_words(self.vectors, positives=[sim_doc_id], topn=word_top_n,
#                                                               feature_to_use=None, print_results=False)
#                 print(len(sim_doc_words))
#                 print(sim_words_relaxed)
#                 print(sim_doc_words)
#                 sim_doc_words_relaxed = set([word for word, sim in sim_doc_words])
#                 sim_doc_words_strict = set([word for word in sim_doc_words_relaxed if word in facet_words])
#                 # print()
#                 shared_words_relaxed = sim_words_relaxed.intersection(sim_doc_words_relaxed)
#                 shared_words_strict = sim_words_strict.intersection(sim_doc_words_strict)
#                 print(len(sim_words_strict), len(sim_doc_words), len(shared_words_relaxed), len(shared_words_strict))
#                 print(sim_words_relaxed)
#                 print(sim_doc_words_relaxed)
#                 print(sim_words_strict)
#                 print(sim_doc_words_strict)
#                 print()
#                 shared_word_values_relaxed.append(len(shared_words_relaxed) / word_top_n)
#                 shared_word_values_strict.append(len(shared_words_strict) / word_top_n)
#                 reciprocal_ranks.append(1/r)
#                 r += 1
#                 # print(len(shared_words), shared_words)
#
#             reciprocal_ranks = [rank / sum(reciprocal_ranks) for rank in reciprocal_ranks]
#             score_relaxed = sum([shared_val * rank_val
#                                  for shared_val, rank_val in zip(shared_word_values_relaxed, reciprocal_ranks)])
#             score_strict = sum([shared_val * rank_val
#                                 for shared_val, rank_val in zip(shared_word_values_strict, reciprocal_ranks)])
#             # print(score)
#             eval_scores_relaxed.append(score_relaxed)
#             eval_scores_strict.append(score_strict)
#             document.sentences = None
#             document.doc_entities = None
#         print(sum(eval_scores_relaxed)/len(eval_scores_relaxed), eval_scores_relaxed)
#         print(sum(eval_scores_strict) / len(eval_scores_strict), eval_scores_strict)
#
#         return eval_scores_relaxed, eval_scores_strict


def calculate_facet_scores(data_sets: List[str], vector_names: List[str], facets: List[str]):
    results = []
    for data_set in data_sets:
        corpus = Corpus.fast_load(path=os.path.join('corpora', data_set), load_entities=False)
        start_time = time.time()
        topic_vecs = TopicModeller.get_topic_distribution(corpus, data_set)
        for vector_name in tqdm(vector_names, desc="Iterate through embedding types", total=len(vector_names)):
            print('---')
            vec_path = Vectorization.build_vec_file_name("all",
                                                         "no_limit",
                                                         data_set,
                                                         "no_filter",
                                                         vector_name,
                                                         "real")

            vecs = Vectorization.my_load_doc2vec_format(vec_path)

            fee = FacetEfficientEvaluation(vectors=vecs, corpus=corpus, data_set_name=data_set, facet_names=facets,
                                           topic_vectors=topic_vecs)
            fac_relaxed_scores, fac_strict_scores = fee.evaluate(word_top_n=1000)

            for fac_name in facets:
                results.append(
                    (data_set, vector_name, fac_name, fac_relaxed_scores[fac_name], fac_strict_scores[fac_name]))

        tuples = []
        for result in results:
            data_set, vector_name, fac_name, relaxed_scores, strict_scores = result
            tuples.append((data_set, fac_name, vector_name,
                           sum(relaxed_scores) / len(relaxed_scores), sum(strict_scores) / len(strict_scores)))

        df = pd.DataFrame(tuples, columns=['Corpus', 'Facet', 'Algorithm', 'Relaxed Score', 'Strict Score'])
        df = df.sort_values(['Corpus', 'Facet', 'Algorithm'])
        print(df)
        df.to_csv('results/facet_evaluation/facet_task_results.csv', index=False)

        results = []
        a_time = time.time() - start_time
        start_time = time.time()

        # for vector_name in tqdm(vector_names, desc="Iterate through embedding types", total=len(vector_names)):
        #     print('---')
        #     vec_path = Vectorizer.build_vec_file_name("all",
        #                                               "no_limit",
        #                                               data_set,
        #                                               "no_filter",
        #                                               vector_name,
        #                                               "real")
        #
        #     vecs = Vectorizer.my_load_doc2vec_format(vec_path)
        #
        #     for fac_name in tqdm(facets, total=len(facets), desc="Iterate through facetes"):
        #         fe = FacetEvaluation(fac_name, vecs, c, data_set)
        #         relaxed_scores, strict_scores = fe.evaluate()
        #         results.append((data_set, vector_name, fac_name, relaxed_scores, strict_scores))
        #
        # tuples = []
        # for result in results:
        #     data_set, vector_name, fac_name, relaxed_scores, strict_scores = result
        #     tuples.append((data_set, vector_name, fac_name,
        #                    sum(relaxed_scores) / len(relaxed_scores), sum(strict_scores) / len(strict_scores)))
        #
        # df = pd.DataFrame(tuples, columns=['Corpus', 'Algorithm', 'Facet', 'Relaxed Score', 'Strict Score'])
        # print(df)
        # df.to_csv('results/facet_evaluation/facet_task_results.csv', index=False)

        b_time = time.time() - start_time
        print(a_time, b_time)


if __name__ == '__main__':
    data_set_names = [
        'classic_gutenberg'
    ]
    algorithms = [
        'book2vec',
        'book2vec_adv',
        'doc2vec',
        'avg_wv2doc'
    ]
    facet_names = [
        # "loc",
        # "time",
        # "atm",
        # "sty",
        "cont"
    ]

    calculate_facet_scores(data_set_names, algorithms, facet_names)
    # data_set_name = "classic_gutenberg"
    # c = Corpus.load_corpus_from_dir_format(os.path.join(f"corpora/{data_set_name}"))
    # # d = TopicModeller.train_lda(c)
    # TopicModeller.get_topic_distribution(c, data_set_name)
