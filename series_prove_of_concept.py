from collections import defaultdict
from typing import Union, Dict
from gensim.models import Doc2Vec
from scipy.stats import stats
from statsmodels.sandbox.stats.multicomp import TukeyHSDResults
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from doc2vec_structures import DocumentKeyedVectors
from utils import DataHandler, Corpus, Preprocesser, Utils
from vectorization import Vectorizer
import random
import pandas as pd
import numpy as np


class Evaluation:
    @staticmethod
    def series_eval(vectors: Union[Doc2Vec, DocumentKeyedVectors],
                    series_dictionary: Dict[str, list],
                    corpus: Corpus,
                    sample_size: int = 50,
                    seed: int = 42,
                    topn: int = 10):
        reverted = Utils.revert_dictionaried_list(series_dictionary)

        doctags = vectors.docvecs.doctags.keys()
        doctags = set([doctag for doctag in doctags if doctag[-1].isdigit() or doctag.endswith('_sum')])
        # print(doctags)
        random.seed(seed)
        sample = random.sample(doctags, sample_size)
        results = []
        soft_results = []
        for doc_id in sample:
            sim_documents = Vectorizer.most_similar_documents(vectors, corpus,
                                                              positives=[doc_id],
                                                              feature_to_use="_sum",
                                                              topn=topn,
                                                              print_results=False)
            hard_correct = 0
            soft_correct = 0
            # print(reverted)
            for sim_doc_id, sim in sim_documents:
                # doc_id.replace("_sum", "")
                replaced_doc_id = doc_id.replace("_sum", "")
                replaced_sim_doc_id = sim_doc_id.replace("_sum", "")
                if reverted[replaced_doc_id] == reverted[replaced_sim_doc_id]:
                    hard_correct += 1

                if corpus.documents[replaced_doc_id].authors == corpus.documents[replaced_sim_doc_id].authors:
                    soft_correct += 1
            hard_correct = hard_correct / len(sim_documents)
            soft_correct = soft_correct / len(sim_documents)
            results.append(hard_correct)
            soft_results.append(soft_correct)

        # mean = sum(results) / len(results)
        # soft_score = sum(soft_results) / len(results)
        # print(f'Scores (h|s){mean} | {soft_score}')
        return np.array(results)

    @staticmethod
    def mean(results: np.ndarray):
        return sum(results) / len(results), np.std(results)

    @staticmethod
    def median(results: np.ndarray):
        return np.median(results), stats.iqr(results)

    @staticmethod
    def one_way_anova(list_results: Dict[str, np.ndarray]):
        vals = [values for values in list_results.values()]
        f, p = stats.f_oneway(*vals)
        significance_dict = defaultdict(str)
        tuples = []
        for group, values in list_results.items():
            for value in values:
                tuples.append((group, value))
            # print(group, Evaluation.mean(values))
        df = pd.DataFrame(tuples, columns=['Group', 'Value'])
        m_comp: TukeyHSDResults = pairwise_tukeyhsd(endog=df['Value'], groups=df['Group'], alpha=0.05)
        m_comp_data = m_comp.summary().data
        mcomp_df = pd.DataFrame(m_comp_data[1:], columns=m_comp_data[0])
        for i, row in mcomp_df.iterrows():
            if row['reject'] and p < 0.05:
                significance_dict[row['group1']] += row['group2'][0]
                significance_dict[row['group2']] += row['group1'][0]
            else:
                significance_dict[row['group1']] += ""
                significance_dict[row['group2']] += ""

        # print(f, p)

        return significance_dict

    @staticmethod
    def t_test(list_results: Dict[str, np.ndarray]):
        outer_dict = {}
        for algorithm_1, values_1 in list_results.items():
            inner_dict = {}
            for algorithm_2, values_2 in list_results.items():
                t, p = stats.ttest_ind(values_1, values_2)
                # print(algorithm_1, algorithm_2, t, p)

                if values_1.mean() > values_2.mean():
                    ba = ">"
                elif values_1.mean() == values_2.mean():
                    ba = "="
                else:
                    ba = "<"

                if p < 0.05:
                    inner_dict[algorithm_2] = f"s{ba}"
                else:
                    inner_dict[algorithm_2] = f"n{ba}"
            outer_dict[algorithm_1] = inner_dict

        return outer_dict


class EvaluationUtils:
    @staticmethod
    def build_paper_table(cache_df: pd.DataFrame, out_path: str) -> pd.DataFrame:
        df_table = cache_df
        df_table["Dataset"] = pd.Categorical(df_table["Dataset"],
                                             categories=df_table["Dataset"].unique(), ordered=True)
        df_table["Algorithm"] = pd.Categorical(df_table["Algorithm"],
                                               categories=df_table["Algorithm"].unique(), ordered=True)
        df_table = df_table.set_index(['Dataset', 'Algorithm'])
        df_table["Filter"] = pd.Categorical(df_table["Filter"], categories=df_table["Filter"].unique(), ordered=True)
        df_table = df_table.pivot(columns='Filter')['Score']
        df_table.to_csv(out_path, index=True, encoding="utf-8")
        return df_table


# class ComplexEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if hasattr(obj, 'json_representation'):
#             return obj.json_representation()
#         else:
#             return json.JSONEncoder.default(self, obj)


def prove_of_concept():
    data_sets = [
        # "summaries",
        "german_books",
        # "litrec",

    ]
    filters = [
        "no_filter",
        # "named_entities",
        "common_words",
        # "nouns",
        # "verbs",
        # "adjectives",
        # "avn"
    ]
    vectorization_algorithms = [
        "avg_wv2doc",
        "doc2vec",
        "book2vec"
    ]
    # dataset_results = {}
    # mean = 0
    tuples = []
    for data_set in data_sets:

        annotated_corpus_path = f'corpora/{data_set}.json'
        annotated_series_corpus_path = f'corpora/{data_set}_series.json'

        number_of_subparts = 10
        # Corpus +Document
        try:
            Corpus(annotated_series_corpus_path)
        except FileNotFoundError:
            try:
                corpus = Corpus(annotated_corpus_path)
                corpus, _ = corpus.fake_series(number_of_sub_parts=number_of_subparts)
            except FileNotFoundError:
                corpus = DataHandler.load_corpus(data_set)
                corpus = Preprocesser.annotate_corpus(corpus[:100])
                corpus.save_corpus(annotated_corpus_path)
                corpus, _ = corpus.fake_series(number_of_sub_parts=number_of_subparts)
                corpus.save_corpus(annotated_series_corpus_path)

        # Series:
        # actual:
        # series_dict = manual_dict
        # corpus = corpus
        # fake:
        # series_dict: {doc_id} -> {series_id}, series_reverse_dict: {series_id} -> [doc_id]
        # filter_results = {}
        for filter_mode in filters:
            # Document-Filter: No, Common Words Del., NER Del., Nouns Del., Verbs Del., ADJ Del., Stopwords Del.
            # common_words: {doc_id} -> [common_words]
            del corpus
            corpus = Corpus(annotated_series_corpus_path)
            common_words_dict = corpus.get_common_words(corpus.series_dict)
            corpus = corpus.filter(mode=filter_mode, common_words=common_words_dict)
            # vectorization_results = {}
            list_results = {}
            # results
            for vectorization_algorithm in vectorization_algorithms:
                vecs = Vectorizer.algorithm(input_str=vectorization_algorithm, corpus=corpus)
                # Scoring:
                results = Evaluation.series_eval(vecs, corpus.series_dict, corpus, topn=number_of_subparts)
                list_results[vectorization_algorithm] = results

            # Evaluation.t_test(list_results)
            significance_dict = Evaluation.one_way_anova(list_results)

            # Scoring
            for vectorization_algorithm in vectorization_algorithms:
                results = list_results[vectorization_algorithm]
                sig = significance_dict[vectorization_algorithm]
                score, deviation = Evaluation.mean(results)
                # vectorization_results[vectorization_algorithm] = score, deviation
                tuples.append((data_set, vectorization_algorithm, filter_mode, f'{score:.4f} ± {deviation:.4f} [{sig}]',
                               Evaluation.median(results)))

    #         filter_results[filter_mode] = vectorization_results
    #     dataset_results[data_set] = filter_results
    # print(dataset_results)

    df = pd.DataFrame(tuples, columns=['Dataset', 'Algorithm', 'Filter', 'Score', 'Median'])
    print(df)
    df.to_csv("/results/cache_table.csv", index=False)
    print(EvaluationUtils.build_paper_table(df, "results/series_experiment_table.csv"))

# check corpus serialization:        X
# check vector serialization:        X
# check evaluation scorer (soft):    X


# Embedding: Avg vec, doc2vec, simpleAspects, simpleSegments, simple A+S


# Algorithm         | Plain | Common Words Del. | NER Del.  | Nouns Del.    | Verbs Del.    | ADJ Del.  | Stopwords Del.
# Avg vec           |   x   |                   |           |               |               |           |
# + simpleAspects   |       |                   |           |               |               |           |
# + simpleSegments  |       |                   |           |               |               |           |
# + simple A + S    |       |                   |           |               |               |           |
# doc2vec           |       |                   |           |               |               |           |
# + simpleAspects   |       |                   |           |               |               |           |
# + simpleSegments  |       |                   |           |               |               |           |
# + simple A + S    |       |                   |           |               |               |           |
# x = Number of correctly identified serial book clusters compared to ...

if __name__ == '__main__':
    prove_of_concept()
