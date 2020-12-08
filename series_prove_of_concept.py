import logging
import logging.config
import multiprocessing
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, Dict, Set, List
from gensim.models import Doc2Vec
from joblib import Parallel, delayed
from scipy.stats import stats
from sklearn import metrics
from statsmodels.sandbox.stats.multicomp import TukeyHSDResults
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tqdm import tqdm

from doc2vec_structures import DocumentKeyedVectors
from utils import DataHandler, Corpus, Preprocesser, Utils, ConfigLoader
from vectorization import Vectorizer
import random
import pandas as pd
import numpy as np


class EvaluationMath:
    @staticmethod
    def mean(results: Union[np.ndarray, List[Union[float, int]]], std: bool = True):
        if not isinstance(results, np.ndarray):
            results = np.array(results)
        # assert np.mean(results) == sum(results) / len(results)
        if std:
            return np.mean(results), np.std(results)
        else:
            return np.mean(results)

    @staticmethod
    def median(results: Union[np.ndarray, List[Union[float, int]]], std: bool = True):
        if not isinstance(results, np.ndarray):
            results = np.array(results)
        if std:
            return np.median(results), stats.iqr(results)
        else:
            return np.median(results)

    @staticmethod
    def one_way_anova(list_results: Dict[str, np.ndarray]):
        def replace_sig_indicator(inp: str):
            if len(inp) > 0:
                inp = f'{",".join([str(i) for i in sorted([int(s) for s in inp.split(",")])])}'
            return inp

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
        group_id_lookup = {key: i + 1 for i, key in enumerate(list_results.keys())}
        for i, row in mcomp_df.iterrows():
            if row['reject'] and p < 0.05:
                g1_commata = ''
                g2_commata = ''
                if len(significance_dict[row['group1']]) > 0:
                    g1_commata = ','
                if len(significance_dict[row['group2']]) > 0:
                    g2_commata = ','
                significance_dict[row['group1']] += f"{g1_commata}{group_id_lookup[row['group2']]}"
                significance_dict[row['group2']] += f"{g2_commata}{group_id_lookup[row['group1']]}"
            else:
                significance_dict[row['group1']] += ""
                significance_dict[row['group2']] += ""

        # print(f, p)
        significance_dict = {key: replace_sig_indicator(value) for key, value in significance_dict.items()}
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


class EvaluationTask(ABC):
    def __init__(self, reverted: Dict[str, str], corpus: Corpus):
        self.reverted = reverted
        self.corpus = corpus

    @abstractmethod
    def has_passed(self, doc_id: str, sim_doc_id: str):
        pass

    @abstractmethod
    def nr_of_possible_matches(self, doc_id: str):
        pass

    def __str__(self):
        return self.__class__.__name__

    __repr__ = __str__

    @staticmethod
    def create_from_name(task_name: str, reverted: Dict[str, str], corpus: Corpus):
        if task_name.lower() == "seriestask" or task_name.lower() == "series_task" or task_name.lower() == "series":
            return SeriesTask(reverted, corpus)
        elif task_name.lower() == "authortask" or task_name.lower() == "author_task" or task_name.lower() == "author":
            return AuthorTask(reverted, corpus)
        elif task_name.lower() == "genretask" or task_name.lower() == "genre_task" or task_name.lower() == "genre":
            return GenreTask(reverted, corpus)
        else:
            raise UserWarning(f"{task_name} is not defined as task")


class SeriesTask(EvaluationTask):
    def has_passed(self, doc_id: str, sim_doc_id: str):
        return self.reverted[doc_id] == self.reverted[sim_doc_id]

    def nr_of_possible_matches(self, doc_id: str):
        return len(self.corpus.series_dict[self.reverted[doc_id]]) - 1


class AuthorTask(EvaluationTask):
    def has_passed(self, doc_id: str, sim_doc_id: str):
        return self.corpus.documents[doc_id].authors == self.corpus.documents[sim_doc_id].authors

    def nr_of_possible_matches(self, doc_id: str):
        return len(self.corpus.get_other_doc_ids_by_same_author(doc_id)) - 1


class GenreTask(EvaluationTask):
    def has_passed(self, doc_id: str, sim_doc_id: str):
        return self.corpus.documents[doc_id].authors == self.corpus.documents[sim_doc_id].authors

    def nr_of_possible_matches(self, doc_id: str):
        return len(self.corpus.get_other_doc_ids_by_same_genres(doc_id)) - 1


class EvaluationMetric:
    @staticmethod
    def precision(sim_documents, doc_id: str, task: EvaluationTask,
                  ignore_same: bool = False, k: int = None):
        # how many selected items are relevant?
        hard_correct = 0
        # soft_correct = 0
        # print(reverted)

        if k is None:
            k = len(sim_documents)

        if doc_id[-1].isalpha():
            doc_id = '_'.join(doc_id.split('_')[:-1])

        for c, (sim_doc_id, _) in enumerate(sim_documents):
            if c == k + 1:
                break

            if sim_doc_id[-1].isalpha():
                sim_doc_id = '_'.join(sim_doc_id.split('_')[:-1])

            if not ignore_same or doc_id != sim_doc_id:
                if task.has_passed(doc_id, sim_doc_id):
                    hard_correct += 1
            # print(task, c, k, hard_correct, doc_id, sim_doc_id)

        hard_correct = hard_correct / k
        return hard_correct

    @staticmethod
    def length_metric(sim_documents, doc_id: str, task: EvaluationTask,
                      ignore_same: bool = False, k: int = None):

        differences = []
        if doc_id[-1].isalpha():
            doc_id = '_'.join(doc_id.split('_')[:-1])

        doc_len = len(task.corpus.documents[doc_id].get_flat_document_tokens())

        for c, (sim_doc_id, _) in enumerate(sim_documents):
            if sim_doc_id[-1].isalpha():
                sim_doc_id = '_'.join(sim_doc_id.split('_')[:-1])

            if not ignore_same or doc_id != sim_doc_id:
                sim_doc_len = len(task.corpus.documents[sim_doc_id].get_flat_document_tokens())
                differences.append(abs(doc_len - sim_doc_len) / doc_len)
            # print(task, c, k, hard_correct, doc_id, sim_doc_id)

        mape = sum(differences) / len(differences) * 100
        return mape

    @staticmethod
    def fair_precision(sim_documents, doc_id: str, task: EvaluationTask,
                       ignore_same: bool = False, k: int = None):
        # how many selected items are relevant?
        if task.nr_of_possible_matches(doc_id) == 0:
            print('zero devision fix at fair_precision')
            return 1
        hard_correct = 0
        # soft_correct = 0
        # print(reverted)
        if k is None:
            k = len(sim_documents)
        if doc_id[-1].isalpha():
            doc_id = '_'.join(doc_id.split('_')[:-1])
        for c, (sim_doc_id, _) in enumerate(sim_documents):
            if c == k + 1:
                break

            if sim_doc_id[-1].isalpha():
                sim_doc_id = '_'.join(sim_doc_id.split('_')[:-1])

            if not ignore_same or doc_id != sim_doc_id:
                if task.has_passed(doc_id, sim_doc_id):
                    hard_correct += 1
            # if corpus.documents[doc_id].authors == corpus.documents[sim_doc_id].authors:
            #     soft_correct += 1

        if k > task.nr_of_possible_matches(doc_id):
            k = task.nr_of_possible_matches(doc_id)

        hard_correct = hard_correct / k
        # soft_correct = soft_correct / len(sim_documents)
        return hard_correct

    @staticmethod
    def recall(sim_documents, doc_id: str, task: EvaluationTask,
               ignore_same: bool = False, k: int = None):
        # how many relevant items are selected?
        if task.nr_of_possible_matches(doc_id) == 0:
            print('zero devision fix at recall')
            return 1
        hard_correct = 0
        # soft_correct = 0
        # print(reverted)
        if k is None:
            k = len(sim_documents)

        if doc_id[-1].isalpha():
            doc_id = '_'.join(doc_id.split('_')[:-1])

        for c, (sim_doc_id, _) in enumerate(sim_documents):
            if c == k + 1:
                break

            if sim_doc_id[-1].isalpha():
                sim_doc_id = '_'.join(sim_doc_id.split('_')[:-1])

            if not ignore_same or doc_id != sim_doc_id:
                if task.has_passed(doc_id, sim_doc_id):
                    hard_correct += 1
            # if corpus.documents[doc_id].authors == corpus.documents[sim_doc_id].authors:
            #     soft_correct += 1

        hard_correct = hard_correct / task.nr_of_possible_matches(doc_id)
        # soft_correct = soft_correct / len(sim_documents)
        return hard_correct

    @staticmethod
    def ap(sim_documents, doc_id: str, task: EvaluationTask,
           ignore_same: bool = False, k: int = None):
        # print(reverted)

        k = 1
        prec_values_at_k = []
        correct_ones = []
        if doc_id[-1].isalpha():
            doc_id = '_'.join(doc_id.split('_')[:-1])
        for sim_doc_id, _ in sim_documents:
            if sim_doc_id[-1].isalpha():
                sim_doc_id = '_'.join(sim_doc_id.split('_')[:-1])

            if not ignore_same or doc_id != sim_doc_id:
                if task.has_passed(doc_id, sim_doc_id):
                    correct_ones.append(k)
                    prec_values_at_k.append(len(correct_ones)/k)

                k += 1
        if len(prec_values_at_k) > 0:
            ap = sum(prec_values_at_k)/len(prec_values_at_k)
        else:
            ap = 0
        return ap

    @staticmethod
    def mrr(sim_documents, doc_id: str, task: EvaluationTask,
            ignore_same: bool = False, k: int = None):
        c = 1
        if doc_id[-1].isalpha():
            doc_id = '_'.join(doc_id.split('_')[:-1])
        for sim_doc_id, _ in sim_documents:
            if sim_doc_id[-1].isalpha():
                sim_doc_id = '_'.join(sim_doc_id.split('_')[:-1])

            if not ignore_same or doc_id != sim_doc_id:
                if task.has_passed(doc_id, sim_doc_id):
                    return 1/c
                c += 1
        return 0

    @staticmethod
    def ndcg(sim_documents, doc_id: str, task: EvaluationTask,
             ignore_same: bool = False, k: int = None):
        # print(task, doc_id, task.corpus.get_other_doc_ids_by_same_author(doc_id), task.nr_of_possible_matches(doc_id))
        if task.nr_of_possible_matches(doc_id) == 0:
            print('zero devision fix at ndcg')
            return 1
        # print(reverted)
        ground_truth_values = []
        predicted_values = []

        replaced_doc_id = doc_id
        if doc_id[-1].isalpha():
            replaced_doc_id = '_'.join(doc_id.split('_')[:-1])
        for c, (sim_doc_id, sim) in enumerate(sim_documents):
            replaced_sim_doc_id = sim_doc_id
            if sim_doc_id[-1].isalpha():
                replaced_sim_doc_id = '_'.join(sim_doc_id.split('_')[:-1])
            # print(doc_id, sim_doc_id, replaced_doc_id, replaced_sim_doc_id, sim)
            if not ignore_same or replaced_doc_id != replaced_sim_doc_id:
                # print('matches:', task.nr_of_possible_matches(doc_id))
                # if c <= task.nr_of_possible_matches(doc_id):
                if sum(ground_truth_values) < task.nr_of_possible_matches(doc_id):
                    ground_truth_values.append(1)
                else:
                    ground_truth_values.append(0)
                if task.has_passed(replaced_doc_id, replaced_sim_doc_id):
                    predicted_values.append(1)
                else:
                    predicted_values.append(0)
            else:
                if c != 0:
                    print(f'First match ({c}) is not lookup document {doc_id} but {sim_doc_id}!')
                    # raise UserWarning(f'First match ({c}) is not lookup document {doc_id} but {sim_doc_id}!')

            # print(task, doc_id, sim_doc_id, predicted_values, ground_truth_values,
            #       task.nr_of_possible_matches(doc_id),
            #       task.corpus.get_other_doc_ids_by_same_author(doc_id),
            #       task.corpus.series_dict[task.reverted[doc_id]])
        assert sum(ground_truth_values) == task.nr_of_possible_matches(doc_id)
        # print(doc_id, ground_truth_values, predicted_values)
        ndcg = metrics.ndcg_score(np.array([ground_truth_values]),
                                  np.array([predicted_values]))
        return ndcg

    @staticmethod
    def f1(sim_documents, doc_id: str, task: EvaluationTask,
           ignore_same: bool = False, k: int = None):
        precision = EvaluationMetric.precision(sim_documents, doc_id,
                                               task, ignore_same, k=k)

        recall = EvaluationMetric.recall(sim_documents, doc_id,
                                         task, ignore_same, k=k)
        f_sum = (precision + recall)
        if f_sum == 0:
            f_sum = 1
        return 2 * (precision * recall) / f_sum

    @staticmethod
    def multi_metric(sim_documents, doc_id: str, task: EvaluationTask,
                     ignore_same: bool = False, k: int = None):
        metric_dict = {
            "prec": EvaluationMetric.precision(sim_documents, doc_id,
                                               task, ignore_same),
            "prec01": EvaluationMetric.precision(sim_documents, doc_id,
                                                 task, ignore_same, k=1),
            "prec03": EvaluationMetric.precision(sim_documents, doc_id,
                                                 task, ignore_same, k=3),
            "prec05": EvaluationMetric.precision(sim_documents, doc_id,
                                                 task, ignore_same, k=5),
            "prec10": EvaluationMetric.precision(sim_documents, doc_id,
                                                 task, ignore_same, k=10),
            "f_prec": EvaluationMetric.fair_precision(sim_documents, doc_id,
                                                      task, ignore_same),
            "f_prec01": EvaluationMetric.fair_precision(sim_documents, doc_id,
                                                        task, ignore_same, k=1),
            "f_prec03": EvaluationMetric.fair_precision(sim_documents, doc_id,
                                                        task, ignore_same, k=3),
            "f_prec05": EvaluationMetric.fair_precision(sim_documents, doc_id,
                                                        task, ignore_same, k=5),
            "f_prec10": EvaluationMetric.fair_precision(sim_documents, doc_id,
                                                        task, ignore_same, k=10),
            "rec": EvaluationMetric.recall(sim_documents, doc_id,
                                           task, ignore_same),
            "rec01": EvaluationMetric.recall(sim_documents, doc_id,
                                             task, ignore_same, k=1),
            "rec03": EvaluationMetric.recall(sim_documents, doc_id,
                                             task, ignore_same, k=3),
            "rec05": EvaluationMetric.recall(sim_documents, doc_id,
                                             task, ignore_same, k=5),
            "rec10": EvaluationMetric.recall(sim_documents, doc_id,
                                             task, ignore_same, k=10),
            "f1": EvaluationMetric.f1(sim_documents, doc_id,
                                      task, ignore_same),
            "f101": EvaluationMetric.f1(sim_documents, doc_id,
                                        task, ignore_same, k=1),
            "f103": EvaluationMetric.f1(sim_documents, doc_id,
                                        task, ignore_same, k=3),
            "f105": EvaluationMetric.f1(sim_documents, doc_id,
                                        task, ignore_same, k=5),
            "f110": EvaluationMetric.f1(sim_documents, doc_id,
                                        task, ignore_same, k=10),
            "ndcg": EvaluationMetric.ndcg(sim_documents, doc_id,
                                          task, ignore_same),
            "mrr": EvaluationMetric.mrr(sim_documents, doc_id,
                                        task, ignore_same),
            "ap": EvaluationMetric.ap(sim_documents, doc_id,
                                      task, ignore_same),
            "length_metric": EvaluationMetric.length_metric(sim_documents, doc_id,
                                                            task, ignore_same)
        }
        return metric_dict


class Evaluation:
    evaluation_metric = EvaluationMetric.precision

    @staticmethod
    def sample_fun(doc_ids: Set[str], sample_size: int, series_dict: Dict[str, List[str]] = None,
                   series_sample: bool = False, seed: int = None):
        if seed:
            random.seed(seed)
        if series_sample:
            if series_dict is None:
                raise UserWarning("No series dict defined!")
            series_ids = series_dict.keys()
            sampled_series_ids = random.sample(series_ids, sample_size)
            return [doc_id for series_id in sampled_series_ids for doc_id in series_dict[series_id]]

        else:
            return random.sample(doc_ids, sample_size)

    @classmethod
    def similar_docs_sample_results(cls, vectors, corpus: Corpus, reverted: Dict[str, str],
                                    sample: List[str], topn: int):
        results = []
        for doc_id in sample:
            sim_documents = Vectorizer.most_similar_documents(vectors, corpus,
                                                              positives=[doc_id],
                                                              feature_to_use="NF",
                                                              topn=topn,
                                                              print_results=False)

            task = SeriesTask(reverted=reverted, corpus=corpus)
            hard_correct = cls.evaluation_metric(sim_documents, doc_id, task=task)

            results.append(hard_correct)

        return results

    # @staticmethod
    # def similar_docs_avg(vectors, corpus: Corpus, reverted: Dict[str, str],
    #                      sample: List[str], topn: int):
    #     results = []
    #     soft_results = []
    #     for doc_id in sample:
    #         hard_it_results = []
    #         soft_it_results = []
    #         for i in range(1, topn+1):
    #             sim_documents = Vectorizer.most_similar_documents(vectors, corpus,
    #                                                               positives=[doc_id],
    #                                                               feature_to_use="NF",
    #                                                               topn=topn,
    #                                                               print_results=False)
    #             hard_correct = 0
    #             soft_correct = 0
    #             # print(reverted)
    #             for sim_doc_id, sim in sim_documents:
    #                 if reverted[doc_id] == reverted[sim_doc_id]:
    #                     hard_correct += 1
    #
    #                 if corpus.documents[doc_id].authors == corpus.documents[sim_doc_id].authors:
    #                     soft_correct += 1
    #             hard_correct = hard_correct / len(sim_documents)
    #             soft_correct = soft_correct / len(sim_documents)
    #             hard_it_results.append(hard_correct)
    #             soft_it_results.append(soft_correct)
    #         # print(doc_id, hard_it_results)
    #         results.append(EvaluationMath.mean(hard_it_results, std=False))
    #         soft_results.append(EvaluationMath.mean(soft_it_results, std=False))
    #     # print('>', len(results))
    #     return results, soft_results

    @staticmethod
    def series_eval(vectors: Union[Doc2Vec, DocumentKeyedVectors],
                    series_dictionary: Dict[str, list],
                    corpus: Corpus,
                    sample_size: int = 50,
                    seed: int = 42,
                    topn: int = 10):
        series_sample = True
        reverted = Utils.revert_dictionaried_list(series_dictionary)
        doctags = vectors.docvecs.doctags.keys()
        doctags = set([doctag for doctag in doctags if doctag[-1].isdigit() or doctag.endswith('_sum')])
        # print(doctags)

        sample = Evaluation.sample_fun(doctags, sample_size=sample_size, series_dict=series_dictionary, seed=seed,
                                       series_sample=series_sample)

        results = Evaluation.similar_docs_sample_results(vectors, corpus, reverted, sample, topn)
        # results2, _ = Evaluation.similar_docs_avg(vectors, corpus, reverted, sample, topn)

        # print(results)
        # print(results2)

        # mean = sum(results) / len(results)
        # soft_score = sum(soft_results) / len(results)
        # print(f'Scores (h|s){mean} | {soft_score}')
        return np.array(results)

    @staticmethod
    def series_eval_bootstrap(vectors: Union[Doc2Vec, DocumentKeyedVectors],
                              series_dictionary: Dict[str, list],
                              corpus: Corpus,
                              sample_size: int = 50,
                              nr_bootstraps: int = 10,
                              topn: int = 10,
                              series_sample: bool = True
                              ):
        random.seed(42)
        seeds = random.sample([i for i in range(0, nr_bootstraps*10)], nr_bootstraps)

        if nr_bootstraps == 1:
            return Evaluation.series_eval(vectors, series_dictionary, corpus, sample_size, seeds[0], topn)

        reverted = Utils.revert_dictionaried_list(series_dictionary)
        doctags = set([doctag for doctag in vectors.docvecs.doctags.keys() if doctag[-1].isdigit()])
        # print(doctags)

        # print(seeds)

        bootstrap_results = []
        for seed in seeds:
            sample = Evaluation.sample_fun(doctags, sample_size=sample_size, series_dict=series_dictionary, seed=seed,
                                           series_sample=series_sample)

            results_fast = Evaluation.similar_docs_sample_results(vectors, corpus, reverted, sample, topn)
            # results_avg, _ = Evaluation.similar_docs_avg(vectors, corpus, reverted, sample, topn)
            # print(seed, Evaluation.mean(results_avg))
            if not series_sample:
                assert len(results_fast) == sample_size == len(sample)
            # print('>>', len(results_fast))
            # fix for results_avg?
            bootstrap_results.append(EvaluationMath.mean(results_fast, std=False))
            # print(results_avg)
            # print(bootstrap_results)
            # print(results2)
        assert len(bootstrap_results) == nr_bootstraps
        # print(bootstrap_results)
        # mean = sum(bootstrap_results) / len(bootstrap_results)
        # soft_score = sum(soft_results) / len(bootstrap_results)
        # print(f'Scores (h|s){mean} | {soft_score}')
        return np.array(bootstrap_results)

    @staticmethod
    def series_eval_full_data(vectors: Union[Doc2Vec, DocumentKeyedVectors],
                              series_dictionary: Dict[str, list],
                              corpus: Corpus,
                              topn: int = 10):

        reverted = Utils.revert_dictionaried_list(series_dictionary)
        doctags = vectors.docvecs.doctags.keys()
        doctags = [doctag for doctag in doctags if doctag[-1].isdigit()]
        logging.info(f'{len(doctags)} document ids found')

        results_fast = Evaluation.similar_docs_sample_results(vectors, corpus, reverted, doctags, topn)
        # results_avg, _ = Evaluation.similar_docs_avg(vectors, corpus, reverted, doctags, topn)
        # print(seed, Evaluation.mean(results_avg))

        return np.array(results_fast)


class EvaluationUtils:
    @staticmethod
    def build_paper_table(cache_df: pd.DataFrame, out_path: str) -> pd.DataFrame:
        if isinstance(cache_df, str):
            cache_df = pd.read_csv(cache_df)
        pd.options.mode.chained_assignment = None
        df_table = cache_df
        df_table["Dataset"] = pd.Categorical(df_table["Dataset"],
                                             categories=df_table["Dataset"].unique(), ordered=True)
        df_table["Algorithm"] = pd.Categorical(df_table["Algorithm"],
                                               categories=df_table["Algorithm"].unique(), ordered=True)
        df_table = df_table.set_index(['Series_length', 'Dataset', 'Task', 'Algorithm', 'Metric'])
        df_table["Filter"] = pd.Categorical(df_table["Filter"], categories=df_table["Filter"].unique(), ordered=True)
        df_table = df_table.pivot(columns='Filter')['Score']
        df_table.to_csv(out_path, index=True, encoding="utf-8")
        return df_table

    @staticmethod
    def create_paper_table(simple_df: Union[pd.DataFrame, str], out_path: str, task: str = None,
                           metrics: List[str] = None, filters: List[str] = None, pivot_column: str = "Metric"):
        if isinstance(simple_df, str):
            simple_df = pd.read_csv(simple_df)
        pd.options.mode.chained_assignment = None
        df_table = simple_df
        if task:
            df_table = df_table[df_table["Task"] == task]
        if metrics:
            df_table = df_table[df_table["Metric"].isin(metrics)]
        if filters:
            df_table = df_table[df_table["Filter"].isin(filters)]
        df_table["Dataset"] = pd.Categorical(df_table["Dataset"],
                                             categories=df_table["Dataset"].unique(), ordered=True)
        print(df_table)
        df_table["Algorithm"] = pd.Categorical(df_table["Algorithm"],
                                               categories=df_table["Algorithm"].unique(), ordered=True)
        df_table["Task"] = pd.Categorical(df_table["Task"],
                                          categories=df_table["Task"].unique(), ordered=True)
        df_table["Metric"] = pd.Categorical(df_table["Metric"],
                                            categories=df_table["Metric"].unique(), ordered=True)
        df_table['Filter'] = pd.Categorical(df_table['Filter'],
                                            categories=df_table['Filter'].unique(), ordered=True)
        index_candidates = ['Series_length', 'Dataset', 'Task', 'Filter', 'Algorithm', 'Metric']

        index_candidates.remove(pivot_column)
        df_table = df_table.set_index(index_candidates)

        df_table = df_table.pivot(columns=pivot_column)['Score']
        df_table.to_csv(out_path, index=True, encoding="utf-8")
        return df_table


class EvaluationRun:
    config = ConfigLoader.get_config()
    min_number_of_subparts = 2
    max_number_of_subparts = 10
    corpus_size = 1000
    num_cores = int(0.75*multiprocessing.cpu_count())

    data_sets = [
        # "summaries",
        "goodreads_genres"
        # "tagged_german_books"
        # "german_books",
        # "german_series"
        # "litrec",

    ]
    filters = [
        "no_filter",
        "named_entities",
        "common_words",
        "stopwords",
        "nouns",
        "verbs",
        "adjectives",
        "avn"
    ]
    vectorization_algorithms = [
        "avg_wv2doc",
        "doc2vec",
        "book2vec",
        "book2vec_wo_raw",
        "book2vec_wo_loc",
        "book2vec_wo_time",
        "book2vec_wo_sty",
        "book2vec_wo_atm",
    ]

    @staticmethod
    def sep_vec_calc_eff(corpus, number_of_subparts, corpus_size, data_set, filter_mode,
                         vectorization_algorithm, fake):
        vec_file_name = Vectorizer.build_vec_file_name(number_of_subparts,
                                                       corpus_size,
                                                       data_set,
                                                       filter_mode,
                                                       vectorization_algorithm,
                                                       fake)
        if not os.path.isfile(vec_file_name):
            EvaluationRun.store_vectors_to_parameters(corpus,
                                                      number_of_subparts,
                                                      corpus_size,
                                                      data_set,
                                                      filter_mode,
                                                      vectorization_algorithm,
                                                      fake)

    @staticmethod
    def store_corpus_to_parameters_eff(corpus, number_of_subparts, corpus_size, data_set,
                                       filter_mode, fake):
        filtered_corpus_dir = Corpus.build_corpus_dir(number_of_subparts,
                                                      corpus_size,
                                                      data_set,
                                                      filter_mode,
                                                      fake)
        if not os.path.isdir(filtered_corpus_dir):
            corpus = corpus.filter_on_copy(mode=filter_mode)
            corpus.save_corpus_adv(filtered_corpus_dir)

    @staticmethod
    def store_vectors_to_parameters(corpus, number_of_subparts, corpus_size, data_set, filter_mode,
                                    vectorization_algorithm, fake):
        vec_file_name = Vectorizer.build_vec_file_name(number_of_subparts,
                                                       corpus_size,
                                                       data_set,
                                                       filter_mode,
                                                       vectorization_algorithm,
                                                       fake)
        if not os.path.isfile(vec_file_name):
            Vectorizer.algorithm(input_str=vectorization_algorithm,
                                 corpus=corpus,
                                 save_path=vec_file_name,
                                 filter_mode=filter_mode,
                                 return_vecs=False)
        # else:
        #     logging.info(f'{vec_file_name} already exists, skip')

    @classmethod
    def build_corpora(cls, parallel: bool = False):
        subparts_bar = tqdm(range(cls.min_number_of_subparts, cls.max_number_of_subparts + 1),
                            desc='1 Iterating through subpart')
        for number_of_subparts in subparts_bar:
            subparts_bar.set_description(f'1 Iterating through subpart >{number_of_subparts}<')
            subparts_bar.refresh()

            data_set_bar = tqdm(cls.data_sets, total=len(cls.data_sets), desc="2 Operate on dataset")
            for data_set in data_set_bar:
                data_set_bar.set_description(f'2 Operate on dataset >{data_set}<')
                data_set_bar.refresh()
                annotated_corpus_path = os.path.join(cls.config["system_storage"]["corpora"],
                                                     f'{data_set}_{cls.corpus_size}')
                annotated_series_corpus_path = os.path.join(cls.config["system_storage"]["corpora"],
                                                            f'{data_set}_{number_of_subparts}_'
                                                            f'{cls.corpus_size}_series')
                # Corpus +Document
                try:
                    # check if series corpus exists
                    # corpus = Corpus(annotated_series_corpus_path)
                    corpus = Corpus.fast_load(path=annotated_series_corpus_path)
                except FileNotFoundError:
                    try:
                        # check if general corpus exists
                        # corpus = Corpus(annotated_corpus_path)
                        corpus = Corpus.fast_load(path=annotated_corpus_path)
                        corpus = Preprocesser.filter_too_small_docs_from_corpus(corpus)
                        corpus, _ = corpus.fake_series(number_of_sub_parts=number_of_subparts)
                        corpus.save_corpus_adv(annotated_series_corpus_path)
                    except FileNotFoundError:
                        # load from raw data
                        corpus = DataHandler.load_corpus(data_set)
                        if cls.corpus_size == "no_limit":
                            corpus = Preprocesser.annotate_corpus(corpus)
                        else:
                            corpus = corpus.sample(cls.corpus_size, seed=42)
                            corpus = Preprocesser.annotate_corpus(corpus)
                            # corpus = Preprocesser.annotate_corpus(corpus[:cls.corpus_size])
                        corpus.save_corpus_adv(annotated_corpus_path)
                        corpus = Preprocesser.filter_too_small_docs_from_corpus(corpus)
                        corpus, _ = corpus.fake_series(number_of_sub_parts=number_of_subparts)
                        corpus.save_corpus_adv(annotated_series_corpus_path)
                # Series:
                # actual:
                # series_dict = manual_dict
                # corpus = corpus
                # fake:
                # series_dict: {doc_id} -> {series_id}, series_reverse_dict: {series_id} -> [doc_id]
                # filter_results = {}
                filter_bar = tqdm(cls.filters, total=len(cls.filters), desc="3 Calculate filter_mode results")
                if parallel:
                    Parallel(n_jobs=cls.num_cores)(
                        delayed(EvaluationRun.store_corpus_to_parameters_eff)(corpus,
                                                                              number_of_subparts,
                                                                              cls.corpus_size,
                                                                              data_set,
                                                                              filter_mode,
                                                                              "fake")
                        for filter_mode in filter_bar)
                else:
                    [EvaluationRun.store_corpus_to_parameters_eff(corpus,
                                                                  number_of_subparts,
                                                                  cls.corpus_size,
                                                                  data_set,
                                                                  filter_mode,
                                                                  "fake")
                     for filter_mode in filter_bar]

    @classmethod
    def train_vecs(cls, parallel: bool = False):
        subparts_bar = tqdm(range(cls.min_number_of_subparts, cls.max_number_of_subparts + 1),
                            desc='1 Iterating through subpart')
        for number_of_subparts in subparts_bar:
            subparts_bar.set_description(f'1 Iterating through subpart >{number_of_subparts}<')
            subparts_bar.refresh()

            data_set_bar = tqdm(cls.data_sets, total=len(cls.data_sets), desc="2 Operate on dataset")
            for data_set in data_set_bar:
                data_set_bar.set_description(f'2 Operate on dataset >{data_set}<')
                data_set_bar.refresh()

                filter_bar = tqdm(cls.filters, total=len(cls.filters), desc="3 Apply Filter")
                for filter_mode in filter_bar:
                    filter_bar.set_description(f'3 Apply Filter >{filter_mode}<')
                    filter_bar.refresh()
                    # filtered_corpus_file_name = Corpus.build_corpus_file_name(number_of_subparts,
                    #                                                           cls.corpus_size,
                    #                                                           data_set,
                    #                                                           filter_mode,
                    #                                                           "fake")
                    # corpus = Corpus(filtered_corpus_file_name)
                    corpus = Corpus.fast_load(number_of_subparts,
                                              cls.corpus_size,
                                              data_set,
                                              filter_mode,
                                              "fake")
                    vec_bar = tqdm(cls.vectorization_algorithms, total=len(cls.vectorization_algorithms),
                                   desc="3 Apply Filter")
                    if parallel:
                        Parallel(n_jobs=cls.num_cores)(delayed(EvaluationRun.sep_vec_calc_eff)(corpus,
                                                                                               number_of_subparts,
                                                                                               cls.corpus_size,
                                                                                               data_set,
                                                                                               filter_mode,
                                                                                               vectorization_algorithm,
                                                                                               "fake")
                                                       for vectorization_algorithm in vec_bar)
                    else:
                        [EvaluationRun.sep_vec_calc_eff(corpus,
                                                        number_of_subparts,
                                                        cls.corpus_size,
                                                        data_set,
                                                        filter_mode,
                                                        vectorization_algorithm,
                                                        "fake")
                         for vectorization_algorithm in vec_bar]

    @staticmethod
    def eval_vec_loop_eff(corpus, number_of_subparts, corpus_size, data_set, filter_mode, vectorization_algorithm,
                          nr_bootstraps, sample_size, series_sample, ensure_no_sample):
        vec_path = Vectorizer.build_vec_file_name(number_of_subparts,
                                                  corpus_size,
                                                  data_set,
                                                  filter_mode,
                                                  vectorization_algorithm,
                                                  "fake")

        vecs = Vectorizer.my_load_doc2vec_format(vec_path)
        # Scoring:
        if nr_bootstraps * sample_size < len(vecs.docvecs.doctags) and not ensure_no_sample:
            results = Evaluation.series_eval_bootstrap(vecs, corpus.series_dict, corpus,
                                                       topn=number_of_subparts,
                                                       sample_size=sample_size,
                                                       nr_bootstraps=nr_bootstraps,
                                                       series_sample=series_sample)
        else:
            if not ensure_no_sample:
                logging.info(f'{nr_bootstraps} bootstraps and {sample_size} samples more work '
                             f'as actual data {len(vecs.docvecs.doctags)} < {nr_bootstraps * sample_size}')
            results = Evaluation.series_eval_full_data(vecs, corpus.series_dict, corpus,
                                                       topn=number_of_subparts)
        return number_of_subparts, data_set, filter_mode, vectorization_algorithm, results

    @classmethod
    def run_evaluation(cls, parallel: bool = False):
        nr_bootstraps = 2
        sample_size = 10
        series_sample = True
        ensure_no_sample = True
        tuples = []
        result_dir = "results"
        experiment_table_name = "series_experiment_table"
        final_path = os.path.join(result_dir, f"simple_{experiment_table_name}.csv")
        cache_path = os.path.join(result_dir, f"cache_{experiment_table_name}.csv")
        paper_path = os.path.join(result_dir, f"{experiment_table_name}.csv")

        res = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict())))
        for number_of_subparts in range(cls.min_number_of_subparts, cls.max_number_of_subparts + 1):
            for data_set in cls.data_sets:
                for filter_mode in cls.filters:
                    # corpus_path = Corpus.build_corpus_file_name(number_of_subparts,
                    #                                             cls.corpus_size,
                    #                                             data_set,
                    #                                             filter_mode,
                    #                                             "fake")
                    # corpus = Corpus(corpus_path)
                    corpus = Corpus.fast_load(number_of_subparts,
                                              cls.corpus_size,
                                              data_set,
                                              filter_mode,
                                              "fake")
                    if parallel:
                        tuple_list_results = Parallel(n_jobs=cls.num_cores)(
                            delayed(EvaluationRun.eval_vec_loop_eff)(corpus,
                                                                     number_of_subparts,
                                                                     cls.corpus_size,
                                                                     data_set,
                                                                     filter_mode,
                                                                     vectorization_algorithm,
                                                                     nr_bootstraps,
                                                                     sample_size,
                                                                     series_sample,
                                                                     ensure_no_sample)
                            for vectorization_algorithm in cls.vectorization_algorithms)
                    else:
                        tuple_list_results = [EvaluationRun.eval_vec_loop_eff(corpus,
                                                                              number_of_subparts,
                                                                              cls.corpus_size,
                                                                              data_set,
                                                                              filter_mode,
                                                                              vectorization_algorithm,
                                                                              nr_bootstraps,
                                                                              sample_size,
                                                                              series_sample,
                                                                              ensure_no_sample)
                                              for vectorization_algorithm in cls.vectorization_algorithms]

                    for subpart_nr, data, filt_mod, vec_algo, results in tuple_list_results:
                        res[subpart_nr][data][filt_mod][vec_algo] = results
        # print(res[10]['summaries']['common_words']['book2vec'])
        # print(res)
        for number_of_subparts in range(cls.min_number_of_subparts, cls.max_number_of_subparts + 1):
            for data_set in cls.data_sets:
                for filter_mode in cls.filters:
                    list_results = res[number_of_subparts][data_set][filter_mode]

                    # Evaluation.t_test(list_results)
                    significance_dict = EvaluationMath.one_way_anova(list_results)

                    # Scoring
                    for vectorization_algorithm in cls.vectorization_algorithms:
                        results = list_results[vectorization_algorithm]
                        sig = significance_dict[vectorization_algorithm]
                        score, deviation = EvaluationMath.mean(results)
                        # vectorization_results[vectorization_algorithm] = score, deviation
                        observation = (number_of_subparts, data_set, vectorization_algorithm, filter_mode,
                                       f'{score:.4f} ± {deviation:.4f} [{sig}]',
                                       EvaluationMath.median(results))
                        tuples.append(observation)

                        df_obs = pd.DataFrame([observation],
                                              columns=['Series_length', 'Dataset', 'Algorithm',
                                                       'Filter', 'Score', 'Median'])
                        df_obs.to_csv(cache_path, mode='a', header=(not os.path.exists(cache_path)), index=False)

        df = pd.DataFrame(tuples, columns=['Series_length', 'Dataset', 'Algorithm', 'Filter', 'Score', 'Median'])
        print(df)
        df.to_csv(final_path, index=False)
        print(EvaluationUtils.build_paper_table(df, paper_path))

    # @classmethod
    # def prove_of_concept(cls):
    #     nr_bootstraps = 2
    #     sample_size = 10
    #     series_sample = True
    #     ensure_no_sample = True
    #     tuples = []
    #     result_dir = "results"
    #     experiment_table_name = "series_experiment_table"
    #     final_path = os.path.join(result_dir, f"simple_{experiment_table_name}.csv")
    #     cache_path = os.path.join(result_dir, f"cache_{experiment_table_name}.csv")
    #     paper_path = os.path.join(result_dir, f"{experiment_table_name}.csv")
    #
    #     # dataset_results = {}
    #     # mean = 0
    #     tuples = []
    #
    #     subparts_bar = tqdm(range(cls.min_number_of_subparts, cls.max_number_of_subparts+1),
    #                         desc='1 Iterating through subpart')
    #     for number_of_subparts in subparts_bar:
    #         subparts_bar.set_description(f'1 Iterating through subpart >{number_of_subparts}<')
    #         subparts_bar.refresh()
    #
    #         data_set_bar = tqdm(cls.data_sets, total=len(cls.data_sets), desc="2 Operate on dataset")
    #         for data_set in data_set_bar:
    #             data_set_bar.set_description(f'2 Operate on dataset >{data_set}<')
    #             data_set_bar.refresh()
    #
    #             annotated_corpus_path = os.path.join(cls.config["system_storage"]["corpora"],
    #                                                  f'{data_set}.json')
    #             annotated_series_corpus_path = os.path.join(cls.config["system_storage"]["corpora"],
    #                                                         f'{data_set}_{number_of_subparts}_series.json')
    #             # Corpus +Document
    #             try:
    #                 # check if series corpus exists
    #                 corpus = Corpus(annotated_series_corpus_path)
    #             except FileNotFoundError:
    #                 try:
    #                     # check if general corpus exists
    #                     corpus = Corpus(annotated_corpus_path)
    #                     corpus = Preprocesser.filter_too_small_docs_from_corpus(corpus)
    #                     corpus, _ = corpus.fake_series(number_of_sub_parts=number_of_subparts)
    #                     corpus.save_corpus(annotated_series_corpus_path)
    #                 except FileNotFoundError:
    #                     # load from raw data
    #                     corpus = DataHandler.load_corpus(data_set)
    #                     if cls.corpus_size == "no_limit":
    #                         corpus = Preprocesser.annotate_corpus(corpus)
    #                     else:
    #                         corpus = corpus.sample(cls.corpus_size, seed=42)
    #                         corpus = Preprocesser.annotate_corpus(corpus)
    #                         # corpus = Preprocesser.annotate_corpus(corpus[:cls.corpus_size])
    #                     corpus.save_corpus(annotated_corpus_path)
    #                     corpus = Preprocesser.filter_too_small_docs_from_corpus(corpus)
    #                     corpus, _ = corpus.fake_series(number_of_sub_parts=number_of_subparts)
    #                     corpus.save_corpus(annotated_series_corpus_path)
    #             # Series:
    #             # actual:
    #             # series_dict = manual_dict
    #             # corpus = corpus
    #             # fake:
    #             # series_dict: {doc_id} -> {series_id}, series_reverse_dict: {series_id} -> [doc_id]
    #             # filter_results = {}
    #             filter_bar = tqdm(cls.filters, total=len(cls.filters), desc="3 Calculate filter_mode results")
    #             for filter_mode in filter_bar:
    #                 filter_bar.set_description(f'3 Calculate filter_mode results >{filter_mode}<')
    #                 filter_bar.refresh()
    #                 # Document-Filter: No, Common Words Del., NER Del., Nouns Del., Verbs Del., ADJ Del.,
    #                 Stopwords Del.
    #                 # common_words: {doc_id} -> [common_words]
    #                 del corpus
    #                 corpus = Corpus(annotated_series_corpus_path)
    #                 common_words_dict = corpus.get_common_words(corpus.series_dict)
    #                 corpus.filter(mode=filter_mode, common_words=common_words_dict)
    #                 # vectorization_results = {}
    #                 list_results = {}
    #                 # results
    #                 vec_bar = tqdm(cls.vectorization_algorithms, total=len(cls.vectorization_algorithms),
    #                                desc="4a Apply Embedding ")
    #                 for vectorization_algorithm in vec_bar:
    #                     vec_bar.set_description(f'4a Apply Embeddings >{vectorization_algorithm}<')
    #                     vec_bar.refresh()
    #                     vecs = Vectorizer.algorithm(input_str=vectorization_algorithm,
    #                                                 corpus=corpus,
    #                                                 save_path=Vectorizer.build_vec_file_name(number_of_subparts,
    #                                                                                          cls.corpus_size,
    #                                                                                          data_set,
    #                                                                                          filter_mode,
    #                                                                                          vectorization_algorithm,
    #                                                                                          "fake"),
    #                                                 filter_mode=filter_mode, return_vecs=True)
    #                     # Scoring:
    #                     if nr_bootstraps * sample_size < len(vecs.docvecs.doctags) and not ensure_no_sample:
    #                         results = Evaluation.series_eval_bootstrap(vecs, corpus.series_dict, corpus,
    #                                                                    topn=number_of_subparts,
    #                                                                    sample_size=sample_size,
    #                                                                    nr_bootstraps=nr_bootstraps,
    #                                                                    series_sample=series_sample)
    #                     else:
    #                         if not ensure_no_sample:
    #                             logging.info(f'{nr_bootstraps} bootstraps and {sample_size} samples more work as act'
    #                                          f'ual data {len(vecs.docvecs.doctags)} < {nr_bootstraps * sample_size}')
    #                         results = Evaluation.series_eval_full_data(vecs, corpus.series_dict, corpus,
    #                                                                    topn=number_of_subparts)
    #                     # print('vec results', len(results))
    #                     list_results[vectorization_algorithm] = results
    #
    #                 # Evaluation.t_test(list_results)
    #                 significance_dict = EvaluationMath.one_way_anova(list_results)
    #
    #                 # Scoring
    #                 vec_eval_bar = tqdm(cls.vectorization_algorithms, total=len(cls.vectorization_algorithms),
    #                                     desc="4b Evaluate Embeddings", disable=False)
    #                 for vectorization_algorithm in vec_eval_bar:
    #                     vec_eval_bar.set_description(f'4b Evaluate Embeddings >{vectorization_algorithm}<')
    #                     vec_eval_bar.refresh()
    #                     results = list_results[vectorization_algorithm]
    #                     sig = significance_dict[vectorization_algorithm]
    #                     score, deviation = EvaluationMath.mean(results)
    #                     # vectorization_results[vectorization_algorithm] = score, deviation
    #                     observation = (number_of_subparts, data_set, vectorization_algorithm, filter_mode,
    #                                    f'{score:.4f} ± {deviation:.4f} [{sig}]',
    #                                    EvaluationMath.median(results))
    #                     tuples.append(observation)
    #
    #                     df_obs = pd.DataFrame([observation],
    #                                           columns=['Series_length', 'Dataset', 'Algorithm',
    #                                                    'Filter', 'Score', 'Median'])
    #                     df_obs.to_csv(cache_path, mode='a', header=(not os.path.exists(cache_path)), index=False)
    #
    #         #         filter_results[filter_mode] = vectorization_results
    #         #     dataset_results[data_set] = filter_results
    #         # print(dataset_results)
    #
    #     df = pd.DataFrame(tuples, columns=['Series_length', 'Dataset', 'Algorithm', 'Filter', 'Score', 'Median'])
    #     print(df)
    #     df.to_csv(final_path, index=False)
    #     print(EvaluationUtils.build_paper_table(df, paper_path))


class RealSeriesEvaluationRun:
    config = ConfigLoader.get_config()
    num_cores = int(0.75*multiprocessing.cpu_count())

    ignore_same = True
    # evaluation_metric = EvaluationMetric.ap
    # evaluation_metric = EvaluationMetric.precision
    # evaluation_metric = EvaluationMetric.ndcg
    evaluation_metric = EvaluationMetric.multi_metric

    data_sets = [
        "german_series",
        # "dta_series"
    ]
    filters = [
        "no_filter",
        # "named_entities",
        "common_words_strict",
        "common_words_strict_general_words_sensitive",
        "common_words_relaxed",
        "common_words_relaxed_general_words_sensitive",
        "common_words_doc_freq"
        # "stopwords",
        # "nouns",
        # "verbs",
        # "adjectives",
        # "avn"
    ]
    vectorization_algorithms = [
        "avg_wv2doc",
        "doc2vec",
        # "longformer_untuned"
        "book2vec",
        # "book2vec_o_raw",
        # "book2vec_o_loc",
        # "book2vec_o_time",
        # "book2vec_o_sty",
        # "book2vec_o_atm",
        # "book2vec_wo_raw",
        # "book2vec_wo_loc",
        # "book2vec_wo_time",
        # "book2vec_wo_sty",
        # "book2vec_wo_atm",
        # "book2vec_w2v",
        "book2vec_adv",
        # "book2vec_adv_o_raw",
        # "book2vec_adv_o_loc",
        # "book2vec_adv_o_time",
        # "book2vec_adv_o_sty",
        # "book2vec_adv_o_atm",
        # "book2vec_adv_o_plot",
        # "book2vec_adv_o_cont",
        # "book2vec_adv_wo_raw",
        # "book2vec_adv_wo_loc",
        # "book2vec_adv_wo_time",
        # "book2vec_adv_wo_sty",
        # "book2vec_adv_wo_atm",
        # "book2vec_adv_wo_plot",
        # "book2vec_adv_wo_cont",
        # "random_aspect2vec"
        # "avg_wv2doc_untrained",
        # "doc2vec_untrained",
        # "book2vec_untrained",
    ]

    task_names = [
        "SeriesTask",
        "AuthorTask",
        # "GenreTask"
    ]

    @classmethod
    def build_real_series_corpora(cls, parallel: bool = False):
        data_set_bar = tqdm(cls.data_sets, total=len(cls.data_sets), desc="2 Operate on dataset")
        for data_set in data_set_bar:
            data_set_bar.set_description(f'2 Operate on dataset >{data_set}<')
            data_set_bar.refresh()
            annotated_corpus_path = os.path.join(cls.config["system_storage"]["corpora"], data_set)
            try:
                # check if general corpus exists
                # corpus = Corpus(annotated_corpus_path)
                corpus = Corpus.fast_load(path=annotated_corpus_path)
            except FileNotFoundError:
                # load from raw data
                corpus = DataHandler.load_corpus(data_set)
                corpus = Preprocesser.annotate_corpus(corpus)
                # corpus.save_corpus(annotated_corpus_path)
                corpus.save_corpus_adv(annotated_corpus_path)

            filter_bar = tqdm(cls.filters, total=len(cls.filters), desc="3 Calculate filter_mode results")
            if parallel:
                Parallel(n_jobs=cls.num_cores)(
                    delayed(EvaluationRun.store_corpus_to_parameters_eff)(corpus,
                                                                          "all",
                                                                          "no_limit",
                                                                          data_set,
                                                                          filter_mode,
                                                                          "real")
                    for filter_mode in filter_bar)
            else:
                [EvaluationRun.store_corpus_to_parameters_eff(corpus,
                                                              "all",
                                                              "no_limit",
                                                              data_set,
                                                              filter_mode,
                                                              "real")
                 for filter_mode in filter_bar]

    @classmethod
    def train_real_series_vecs(cls, parallel: bool = False):
        data_set_bar = tqdm(cls.data_sets, total=len(cls.data_sets), desc="2 Operate on dataset")
        for data_set in data_set_bar:
            data_set_bar.set_description(f'2 Operate on dataset >{data_set}<')
            data_set_bar.refresh()

            filter_bar = tqdm(cls.filters, total=len(cls.filters), desc="3 Apply Filter")
            for filter_mode in filter_bar:
                filter_bar.set_description(f'3 Apply Filter >{filter_mode}<')
                filter_bar.refresh()
                # filtered_corpus_file_name = Corpus.build_corpus_file_name("all",
                #                                                           "no_limit",
                #                                                           data_set,
                #                                                           filter_mode,
                #                                                           "real")
                # corpus = Corpus(filtered_corpus_file_name)
                corpus = Corpus.fast_load("all",
                                          "no_limit",
                                          data_set,
                                          filter_mode,
                                          "real")

                vec_bar = tqdm(cls.vectorization_algorithms, total=len(cls.vectorization_algorithms),
                               desc="4 Vectorize")
                if parallel:
                    Parallel(n_jobs=cls.num_cores)(delayed(EvaluationRun.sep_vec_calc_eff)(corpus,
                                                                                           "all",
                                                                                           "no_limit",
                                                                                           data_set,
                                                                                           filter_mode,
                                                                                           vec_algorithm,
                                                                                           "real")
                                                   for vec_algorithm in vec_bar)
                else:
                    [EvaluationRun.sep_vec_calc_eff(corpus,
                                                    "all",
                                                    "no_limit",
                                                    data_set,
                                                    filter_mode,
                                                    vectorization_algorithm,
                                                    "real")
                     for vectorization_algorithm in vec_bar]

    @classmethod
    def aggregate_results(cls, data_set: str, task_name: str, metric_name: str, filter_mode: str,
                          results_as_dict: Dict[str, np.ndarray],
                          cache_path: str):
        tuples = []
        significance_dict = EvaluationMath.one_way_anova(results_as_dict)
        vec_bar = tqdm(cls.vectorization_algorithms, total=len(cls.vectorization_algorithms),
                       desc="Store final results for algorithm")
        # Scoring

        for vectorization_algorithm in vec_bar:
            results = results_as_dict[vectorization_algorithm]
            sig = significance_dict[vectorization_algorithm]
            score, deviation = EvaluationMath.mean(results)
            # vectorization_results[vectorization_algorithm] = score, deviation
            observation = ("all", data_set, task_name, metric_name, vectorization_algorithm, filter_mode,
                           f'{score:.4f} ± {deviation:.4f} [{sig}]',
                           EvaluationMath.median(results))
            tuples.append(observation)

            df_obs = pd.DataFrame([observation],
                                  columns=['Series_length', 'Dataset', 'Task', 'Metric', 'Algorithm',
                                           'Filter', 'Score', 'Median'])
            df_obs.to_csv(cache_path, mode='a', header=(not os.path.exists(cache_path)), index=False)
        return tuples

    @classmethod
    def run_evaluation_eff(cls, parallel: bool = False):
        tuples = []
        result_dir = "results"
        experiment_table_name = "series_experiment_table"
        final_path = os.path.join(result_dir, f"simple_{experiment_table_name}.csv")
        cache_path = os.path.join(result_dir, f"cache_{experiment_table_name}.csv")
        paper_path = os.path.join(result_dir, f"{experiment_table_name}.csv")

        res = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict())))
        for data_set in tqdm(cls.data_sets, total=len(cls.data_sets), desc=f"Evaluate datasets"):
            for filter_mode in tqdm(cls.filters, total=len(cls.filters), desc=f"Evaluate filters"):
                # corpus_path = Corpus.build_corpus_file_name("all",
                #                                             "no_limit",
                #                                             data_set,
                #                                             filter_mode,
                #                                             "real")
                # corpus = Corpus(corpus_path)
                corpus = Corpus.fast_load("all",
                                          "no_limit",
                                          data_set,
                                          filter_mode,
                                          "real")
                vec_bar = tqdm(cls.vectorization_algorithms,
                               total=len(cls.vectorization_algorithms),
                               desc=f"Evaluate algorithm")
                if parallel:
                    tuple_list_results = Parallel(n_jobs=cls.num_cores)(
                        delayed(RealSeriesEvaluationRun.eval_vec_loop_eff)(corpus,
                                                                           "all",
                                                                           "no_limit",
                                                                           data_set,
                                                                           filter_mode,
                                                                           vectorization_algorithm)
                        for vectorization_algorithm in vec_bar)
                else:
                    tuple_list_results = [RealSeriesEvaluationRun.eval_vec_loop_eff(corpus,
                                                                                    "all",
                                                                                    "no_limit",
                                                                                    data_set,
                                                                                    filter_mode,
                                                                                    vectorization_algorithm)
                                          for vectorization_algorithm in vec_bar]

                for subpart_nr, data, filt_mod, vec_algo, results in tuple_list_results:
                    res[subpart_nr][data][filt_mod][vec_algo] = results

        for data_set in tqdm(cls.data_sets, total=len(cls.data_sets), desc="Store final results for dataset"):
            for filter_mode in tqdm(cls.filters, total=len(cls.filters), desc="Store final results for filter"):
                list_results = res["all"][data_set][filter_mode]

                if isinstance(list(list_results.values())[0], dict):
                    reverted_nesting = defaultdict(lambda: defaultdict(dict))
                    for algorithm_key, task_dict in list_results.items():
                        for task_name, metric_dict in task_dict.items():
                            for metric_name, metric_res in metric_dict.items():
                                reverted_nesting[task_name][metric_name].update({algorithm_key: metric_res})

                    list_results = reverted_nesting
                    # metric_tuples = defaultdict(list)
                    for task_name, task_dict in list_results.items():
                        for metric, metric_results in task_dict.items():
                            tuples.extend(cls.aggregate_results(data_set, task_name, metric,
                                                                filter_mode, metric_results,
                                                                cache_path))
                    # tuples.extend(metric_tuples)

                else:
                    tuples.extend(cls.aggregate_results(data_set, "Series", cls.evaluation_metric.__name__,
                                                        filter_mode, list_results, cache_path))

        df = pd.DataFrame(tuples, columns=['Series_length', 'Dataset', 'Task', 'Metric', 'Algorithm',
                                           'Filter', 'Score', 'Median'])
        print(df)
        df.to_csv(final_path, index=False)
        print(EvaluationUtils.build_paper_table(df, paper_path))
        # print(EvaluationUtils.create_paper_table(df, paper_path, metrics=["prec", "prec01",
        #                                                                   "prec03", "prec05", "prec10"]))

    @classmethod
    def eval_vec_loop_eff(cls, corpus, number_of_subparts, corpus_size, data_set, filter_mode, vectorization_algorithm):
        vec_path = Vectorizer.build_vec_file_name(number_of_subparts,
                                                  corpus_size,
                                                  data_set,
                                                  filter_mode,
                                                  vectorization_algorithm,
                                                  "real")
        summation_method = "NF"
        try:
            vectors = Vectorizer.my_load_doc2vec_format(vec_path)
        except FileNotFoundError:
            if "_o_" in vectorization_algorithm:
                vec_splitted = vectorization_algorithm.split("_o_")[0]
                focus_facette = vectorization_algorithm.split("_o_")[1]
                base_algorithm = vec_splitted
                vec_path = Vectorizer.build_vec_file_name(number_of_subparts,
                                                          corpus_size,
                                                          data_set,
                                                          filter_mode,
                                                          base_algorithm,
                                                          "real")
                vectors = Vectorizer.my_load_doc2vec_format(vec_path)
                summation_method = focus_facette
            else:
                raise FileNotFoundError
        reverted = Utils.revert_dictionaried_list(corpus.series_dict)
        doctags = vectors.docvecs.doctags.keys()
        doctags = [doctag for doctag in doctags if doctag[-1].isdigit()]

        results = []
        task_results = defaultdict(list)
        # test_results = []
        topn_value = 1
        if cls.ignore_same:
            topn_value = 1

        tasks = [EvaluationTask.create_from_name(task_name, reverted=reverted, corpus=corpus)
                 for task_name in cls.task_names]

        for doc_id in doctags:
            # topn = len(corpus.series_dict[reverted[doc_id]])
            topn = 20
            # todo feature to use = each aspect: x
            #  result merge :x
            #  results for different metrics: x
            #  different tasks: x
            #  common words überarbeiten: x
            #  influence of length: x

            if topn > topn_value:
                # print('#############', doc_id, len(doctags))
                sim_documents = Vectorizer.most_similar_documents(vectors, corpus,
                                                                  positives=[doc_id],
                                                                  feature_to_use=summation_method,
                                                                  topn=topn,
                                                                  print_results=False)

                for task in tasks:
                    metric_results = cls.evaluation_metric(sim_documents, doc_id,
                                                           task,
                                                           ignore_same=cls.ignore_same)
                    task_results[str(task)].append(metric_results)
                    results.append(metric_results)
        # test_results = task_results
        # print('><', results)
        # print('>><', test_results)
        # results_avg, _ = Evaluation.similar_docs_avg(vectors, corpus, reverted, doctags, topn)
        # print(seed, Evaluation.mean(results_avg))
        if isinstance(results[0], dict):
            results = {k: np.array([dic[k] for dic in results]) for k in results[0]}
        else:
            results = np.array(results)

        final_task_results = {}
        for task_name, task_results in task_results.items():
            if isinstance(task_results[0], dict):
                task_results = {k: np.array([dic[k] for dic in task_results]) for k in task_results[0]}
            else:
                task_results = np.array(results)
            final_task_results[task_name] = task_results

        return number_of_subparts, data_set, filter_mode, vectorization_algorithm, final_task_results
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
    # done:
    # check if corpus or vecs already existing before calc in step 1: x
    # parrallize one: x
    # into corpora processing and vector processing
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logging.confi

    # EvaluationRun.build_corpora()
    # EvaluationRun.train_vecs()
    # EvaluationRun.run_evaluation()

    # RealSeriesEvaluationRun.build_real_series_corpora()
    # RealSeriesEvaluationRun.train_real_series_vecs()
    # RealSeriesEvaluationRun.run_evaluation_eff()
    print(EvaluationUtils.create_paper_table("results/simple_series_experiment_table.csv", "results/z_table.csv",
                                             metrics=["ndcg", "prec", "prec01", "prec03", "prec05", "prec10", "length_metric"],
                                             filters=["common_words_relaxed", "common_words_strict",
                                                      "common_words_strict_general_words_sensitive",
                                                      "common_words_relaxed_general_words_sensitive",
                                                      "common_words_doc_freq"]))
