import os
from typing import Union

from gensim.models import Doc2Vec

from corpus_structure import Corpus, Token, Document
from doc2vec_structures import DocumentKeyedVectors
from vectorization import Vectorizer
import pandas as pd


class FacetEvaluation:
    def __init__(self, facet_name: str, vectors: Union[Doc2Vec, DocumentKeyedVectors], corpus: Corpus):
        self.facet_name = facet_name
        self.vectors = vectors
        self.corpus = corpus
        self.same_facet_words = True
        self.lemma = False
        self.lower = False

    def get_facet_words(self, document: Document):
        if self.same_facet_words:
            facet_words = []
            if self.facet_name.lower() == "loc":
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
                facet_words = [word.representation() for word in facet_words]
            elif self.facet_name.lower() == "time":
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
                facet_words = [word.representation() for word in facet_words]
            elif self.facet_name.lower() == "atm":
                facet_words.extend(document.get_flat_and_filtered_document_tokens(lemma=self.lemma,
                                                                                  lower=self.lower,
                                                                                  pos=["ADJ", "ADV"]))
            elif self.facet_name.lower() == "sty":
                facet_words.extend(document.get_flat_and_filtered_document_tokens(lemma=self.lemma,
                                                                                  lower=self.lower,
                                                                                  focus_stopwords=True))
            else:
                raise UserWarning(f"Not supported facet name >{self.facet_name}<!")


        else:
            facet_words = []

        return facet_words

    def evaluate(self):
        doc_top_n = 20
        word_top_n = 100
        scores = []
        for doc_id, document in self.corpus.documents.items():
            # facet_doc_id = f'{doc_id}_{self.facet_name}'
            # print(doc_id)
            document.load_sentences_from_disk()
            document.set_entities()
            # print(document.doc_entities.keys())

            sim_docs = Vectorizer.most_similar_documents(self.vectors, self.corpus, positives=doc_id,
                                                         topn=doc_top_n,
                                                         feature_to_use=self.facet_name, print_results=False)

            sim_words = Vectorizer.most_similar_words(self.vectors, positives=[doc_id],
                                                      topn=word_top_n,
                                                      feature_to_use=self.facet_name, print_results=False)
            facet_words = self.get_facet_words(document)
            sim_words = set([word for word, sim in sim_words
                             if not self.same_facet_words or word in facet_words])
            # print(sim_words)
            # print(facet_doc_id)
            # print(sim_docs, sim_words)
            # print('_____')
            shared_word_values = []
            reciprocal_ranks = []
            r = 1
            for sim_doc_id, sim_doc in sim_docs:
                if str(sim_doc_id).startswith(doc_id):
                    continue
                # print('>>', sim_doc_id)
                sim_doc_words = Vectorizer.most_similar_words(self.vectors, positives=[sim_doc_id], topn=word_top_n,
                                                              feature_to_use=None, print_results=False)

                sim_doc_words = set([word for word, sim in sim_doc_words
                                     if not self.same_facet_words or word in facet_words])
                shared_words = sim_words.intersection(sim_doc_words)
                shared_word_values.append(len(shared_words) / word_top_n)
                reciprocal_ranks.append(1/r)
                r += 1
                # print(len(shared_words), shared_words)

            reciprocal_ranks = [rank / sum(reciprocal_ranks) for rank in reciprocal_ranks]
            score = sum([shared_val * rank_val for shared_val, rank_val in zip(shared_word_values, reciprocal_ranks)])
            # print(score)
            scores.append(score)
            document.sentences = None
            document.doc_entities = None
        print(sum(scores)/len(scores), scores)

        return scores

        # doc_id = f'gs_0_0_{self.facet_name}'
        # print(doc_id)
        # res = Vectorizer.most_similar_words(self.vectors, positives=[doc_id])
        #
        # doc_id = f'gs_0_1_{self.facet_name}'
        # print(doc_id)
        # res = Vectorizer.most_similar_words(self.vectors, positives=[doc_id])
        #
        # doc_id = f'gs_10_0_{self.facet_name}'
        # print(doc_id)
        # res = Vectorizer.most_similar_words(self.vectors, positives=[doc_id])
        #
        # doc_id = f'gs_10_1_{self.facet_name}'
        # print(doc_id)
        # res = Vectorizer.most_similar_words(self.vectors, positives=[doc_id])


if __name__ == '__main__':
    data_set = 'german_series'
    vector_names = [
        'book2vec',
        'doc2vec'
    ]
    facets = [
        "loc",
        "time",
        "atm",
        "sty"
    ]
    c = Corpus.fast_load(path=os.path.join('corpora', data_set), load_entities=False)

    results = []
    for vector_name in vector_names:
        print('---')
        vec_path = Vectorizer.build_vec_file_name("all",
                                                  "no_limit",
                                                  "german_series",
                                                  "no_filter",
                                                  vector_name,
                                                  "real")

        vecs = Vectorizer.my_load_doc2vec_format(vec_path)

        for facet_name in facets:
            fe = FacetEvaluation(facet_name, vecs, c)
            results.append((data_set, vector_name, facet_name, fe.evaluate()))

    tuples = []
    for result in results:
        data_set, vector_name, facet_name, scores = result
        tuples.append((data_set, vector_name, facet_name, sum(scores) / len(scores)))

    df = pd.DataFrame(tuples, columns=['Corpus', 'Algorithm', 'Facet', 'Score'])
    print(df)
    df.to_csv('results/facet_evaluation/facet_task_results.csv', index=False)


