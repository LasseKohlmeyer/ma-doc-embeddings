import json
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
from corpus_structure import Corpus, Utils, ConfigLoader
from document_segments import chunk_documents
from vectorization import Vectorizer
import random
import pandas as pd
import numpy as np

from vectorization_utils import Vectorization
from word_movers_distance import WordMoversDistance


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
        # print(tuples)
        df = pd.DataFrame(tuples, columns=['Group', 'Value'])
        try:
            m_comp: TukeyHSDResults = pairwise_tukeyhsd(endog=df['Value'], groups=df['Group'], alpha=0.05)
        except ValueError:
            list_results.keys()
            return {key: "" for key in list_results.keys()}
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
    def __init__(self, reverted: Dict[str, str], corpus: Corpus, topn):
        self.reverted = reverted
        self.corpus = corpus
        self.topn = topn - 1
        self.correct = []
        self.uncorrect = []
        self.truth = {}

    @abstractmethod
    def has_passed(self, doc_id: str, sim_doc_id: str):
        pass

    @abstractmethod
    def ground_truth(self, doc_id: str):
        pass

    @abstractmethod
    def nr_of_possible_matches(self, doc_id: str):
        pass

    def __str__(self):
        return self.__class__.__name__

    __repr__ = __str__

    def store_passed_results(self, passed, doc_id, sim_doc_id):
        if passed:
            self.correct.append((doc_id, sim_doc_id))
        else:
            self.uncorrect.append((doc_id, sim_doc_id))

        if doc_id not in self.truth:
            self.truth = self.ground_truth(doc_id)


    @staticmethod
    def create_from_name(task_name: str, reverted: Dict[str, str], corpus: Corpus, topn: int):
        if task_name.lower() == "seriestask" or task_name.lower() == "series_task" or task_name.lower() == "series":
            # if corpus.series_dict is None:
            #     raise UserWarning("No series dictionary found for corpus!")
            return SeriesTask(reverted, corpus, topn)
        elif task_name.lower() == "authortask" or task_name.lower() == "author_task" or task_name.lower() == "author":
            return AuthorTask(reverted, corpus, topn)
        elif task_name.lower() == "genretask" or task_name.lower() == "genre_task" or task_name.lower() == "genre":
            return GenreTask(reverted, corpus, topn)
        else:
            raise UserWarning(f"{task_name} is not defined as task")


class SeriesTask(EvaluationTask):
    def ground_truth(self, doc_id):
        return self.corpus.series_dict[self.reverted[doc_id]]

    def has_passed(self, doc_id: str, sim_doc_id: str):
        try:
            passed = self.reverted[doc_id] == self.reverted[sim_doc_id]
            self.store_passed_results(passed, doc_id, sim_doc_id)
            return passed
        except KeyError:
            if sim_doc_id not in self.reverted:
                return False
            else:
                print(doc_id, sim_doc_id, self.reverted)
                raise UserWarning("No proper series handling")
            # return True

    def nr_of_possible_matches(self, doc_id: str):
        try:
            real_matches = len(self.ground_truth(doc_id)) - 1
            if real_matches > self.topn:
                return self.topn
            return real_matches
        except KeyError:
            raise UserWarning("No proper series handling")
            # return 0


class AuthorTask(EvaluationTask):
    def ground_truth(self, doc_id):
        return self.corpus.get_other_doc_ids_by_same_author(doc_id)

    def has_passed(self, doc_id: str, sim_doc_id: str):
        passed = self.corpus.documents[doc_id].authors == self.corpus.documents[sim_doc_id].authors
        self.store_passed_results(passed, doc_id, sim_doc_id)
        return passed

    def nr_of_possible_matches(self, doc_id: str):
        real_matches = len(self.ground_truth(doc_id))
        # print(real_matches, doc_id, self.corpus.get_other_doc_ids_by_same_author(doc_id))
        if real_matches > self.topn:
            return self.topn
        return real_matches


class GenreTask(EvaluationTask):
    def ground_truth(self, doc_id):
        return self.corpus.get_other_doc_ids_by_same_genres(doc_id)

    def has_passed(self, doc_id: str, sim_doc_id: str):
        passed = self.corpus.documents[doc_id].genres == self.corpus.documents[sim_doc_id].genres
        self.store_passed_results(passed, doc_id, sim_doc_id)
        return passed

    def nr_of_possible_matches(self, doc_id: str):
        real_matches = len(self.ground_truth(doc_id))
        # real_matches = len(self.corpus.get_other_doc_ids_by_same_genres(doc_id))
        if real_matches > self.topn:
            return self.topn
        return real_matches


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

        # doc_len = len(task.corpus.documents[doc_id].get_flat_document_tokens())
        doc_len = task.corpus.documents[doc_id].length  # len(task.corpus.documents[doc_id].get_flat_tokens_from_disk())

        for c, (sim_doc_id, _) in enumerate(sim_documents):
            if sim_doc_id[-1].isalpha():
                sim_doc_id = '_'.join(sim_doc_id.split('_')[:-1])

            if not ignore_same or doc_id != sim_doc_id:
                # sim_doc_len = len(task.corpus.documents[sim_doc_id].get_flat_document_tokens())
                sim_doc_len = task.corpus.documents[sim_doc_id].length
                differences.append(abs(doc_len - sim_doc_len) / doc_len)
            # print(task, c, k, hard_correct, doc_id, sim_doc_id)

        mape = sum(differences) / len(differences) * 100
        return mape

    @staticmethod
    def fair_precision(sim_documents, doc_id: str, task: EvaluationTask,
                       ignore_same: bool = False, k: int = None):
        # how many selected items are relevant?
        if task.nr_of_possible_matches(doc_id) == 0:
            # print('zero devision fix at fair_precision')
            return None
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
            # print('zero devision fix at recall')
            return None
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
    def fair_recall(sim_documents, doc_id: str, task: EvaluationTask,
                    ignore_same: bool = False, k: int = None):
        # how many relevant items are selected?
        if task.nr_of_possible_matches(doc_id) == 0:
            # print('zero devision fix at recall')
            return None
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
        relevant_items = task.nr_of_possible_matches(doc_id)
        if k is None:
            k = 100
        if relevant_items > k:
            relevant_items = k

        hard_correct = hard_correct / relevant_items
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
                    prec_values_at_k.append(len(correct_ones) / k)

                k += 1
        if len(prec_values_at_k) > 0:
            ap = sum(prec_values_at_k) / len(prec_values_at_k)
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
                    return 1 / c
                c += 1
        return 0

    @staticmethod
    def ndcg(sim_documents, doc_id: str, task: EvaluationTask,
             ignore_same: bool = False, k: int = None):
        # print(task, doc_id, task.corpus.get_other_doc_ids_by_same_author(doc_id), task.nr_of_possible_matches(doc_id))
        if task.nr_of_possible_matches(doc_id) == 0:
            # print('zero devision fix at ndcg')
            return None, {}, {}
        # print(reverted)
        ground_truth_values = []
        predicted_values = []

        replaced_doc_id = doc_id
        id_annontation = defaultdict(list)
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
                    id_annontation[replaced_doc_id].append((replaced_sim_doc_id, 1))
                else:
                    predicted_values.append(0)
                    id_annontation[replaced_doc_id].append((replaced_sim_doc_id, 0))
            else:
                if c != 0:
                    print(f'First match ({c}) is not lookup document {doc_id} but {sim_doc_id}!')

            # print(task, doc_id, sim_doc_id, predicted_values, ground_truth_values,
            #       task.nr_of_possible_matches(doc_id),
            #       task.corpus.get_other_doc_ids_by_same_author(doc_id),
            #       task.corpus.series_dict[task.reverted[doc_id]])
        # print(ground_truth_values, predicted_values)
        # print(doc_id, sim_documents, sum(ground_truth_values),  task.nr_of_possible_matches(doc_id))

        found_ids = set([found_id[0] for found_id in id_annontation[doc_id]])

        print(doc_id, len(set(task.ground_truth(doc_id))), set(task.ground_truth(doc_id)))
        print(doc_id, len(found_ids), found_ids)
        print(doc_id, len(set(task.ground_truth(doc_id)).difference(found_ids)), set(task.ground_truth(doc_id)).difference(found_ids))
        print()
        missed = {doc_id: list(set(task.ground_truth(doc_id)).difference(found_ids))}
        assert sum(ground_truth_values) == task.nr_of_possible_matches(doc_id)
        # print(doc_id, ground_truth_values, predicted_values)
        ndcg = metrics.ndcg_score(np.array([ground_truth_values]),
                                  np.array([predicted_values]))
        return ndcg, id_annontation, missed

    @staticmethod
    def f1(sim_documents, doc_id: str, task: EvaluationTask,
           ignore_same: bool = False, k: int = None):
        precision = EvaluationMetric.precision(sim_documents, doc_id,
                                               task, ignore_same, k=k)

        recall = EvaluationMetric.recall(sim_documents, doc_id,
                                         task, ignore_same, k=k)
        if precision and recall:
            f_sum = (precision + recall)
        else:
            return None
        if f_sum == 0:
            f_sum = 1
        return 2 * (precision * recall) / f_sum

    @staticmethod
    def fair_f1(sim_documents, doc_id: str, task: EvaluationTask,
                ignore_same: bool = False, k: int = None):
        precision = EvaluationMetric.fair_precision(sim_documents, doc_id,
                                               task, ignore_same, k=k)

        recall = EvaluationMetric.fair_recall(sim_documents, doc_id,
                                              task, ignore_same, k=k)

        if precision and recall:
            f_sum = (precision + recall)
        else:
            return None
        if f_sum == 0:
            f_sum = 1
        # print(precision, recall, 2 * (precision * recall) / f_sum)
        return 2 * (precision * recall) / f_sum

    @staticmethod
    def multi_metric(sim_documents, doc_id: str, task: EvaluationTask,
                     ignore_same: bool = False, k: int = None):
        ndcg, doc_id_dict, missed = EvaluationMetric.ndcg(sim_documents, doc_id,
                                                          task, ignore_same)
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
            "f_rec": EvaluationMetric.fair_recall(sim_documents, doc_id,
                                                  task, ignore_same),
            "f_rec01": EvaluationMetric.fair_recall(sim_documents, doc_id,
                                                    task, ignore_same, k=1),
            "f_rec03": EvaluationMetric.fair_recall(sim_documents, doc_id,
                                                    task, ignore_same, k=3),
            "f_rec05": EvaluationMetric.fair_recall(sim_documents, doc_id,
                                                    task, ignore_same, k=5),
            "f_rec10": EvaluationMetric.fair_recall(sim_documents, doc_id,
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
            "f_f1": EvaluationMetric.fair_f1(sim_documents, doc_id,
                                             task, ignore_same),
            "f_f101": EvaluationMetric.fair_f1(sim_documents, doc_id,
                                               task, ignore_same, k=1),
            "f_f103": EvaluationMetric.fair_f1(sim_documents, doc_id,
                                               task, ignore_same, k=3),
            "f_f105": EvaluationMetric.fair_f1(sim_documents, doc_id,
                                               task, ignore_same, k=5),
            "f_f110": EvaluationMetric.fair_f1(sim_documents, doc_id,
                                               task, ignore_same, k=10),
            "ndcg": ndcg,
            "mrr": EvaluationMetric.mrr(sim_documents, doc_id,
                                        task, ignore_same),
            "ap": EvaluationMetric.ap(sim_documents, doc_id,
                                      task, ignore_same),
            "length_metric": EvaluationMetric.length_metric(sim_documents, doc_id,
                                                            task, ignore_same)
        }
        return metric_dict, doc_id_dict, missed


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
            sim_documents = Vectorization.most_similar_documents(vectors, corpus,
                                                                 positives=[doc_id],
                                                                 feature_to_use="NF",
                                                                 topn=topn,
                                                                 print_results=False,
                                                                 series=True)

            task = SeriesTask(reverted=reverted, corpus=corpus, topn=topn)
            # print(task.nr_of_possible_matches(doc_id))
            # print(doc_id, sim_documents)
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
    #             sim_documents = Vectorization.most_similar_documents(vectors, corpus,
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
        seeds = random.sample([i for i in range(0, nr_bootstraps * 10)], nr_bootstraps)

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
                           used_metrics: List[str] = None, filters: List[str] = None, pivot_column: str = "Metric"):
        if isinstance(simple_df, str):
            simple_df = pd.read_csv(simple_df)
        pd.options.mode.chained_assignment = None
        df_table = simple_df
        if task:
            df_table = df_table[df_table["Task"] == task]
        if used_metrics:
            df_table = df_table[df_table["Metric"].isin(used_metrics)]
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
        print(df_table)
        print(df_table.columns, pivot_column)
        df_table = df_table.pivot(columns=pivot_column)['Score']
        df_table.to_csv(out_path, index=True, encoding="utf-8")
        return df_table

    @staticmethod
    def latex_table(simple_df: Union[pd.DataFrame, str], drop_columns: List[str] = None):
        if isinstance(simple_df, str):
            simple_df = pd.read_csv(simple_df)
        pd.options.mode.chained_assignment = None
        if drop_columns:
            simple_df = simple_df.drop(drop_columns, axis=1)
        print(simple_df.to_latex(index=False))

    @staticmethod
    def attributes_based_on_data(data_set_name: str):
        if "_fake_series" in data_set_name:
            subparts = range(EvalParams.min_number_of_subparts, EvalParams.max_number_of_subparts + 1)
            corpus_size = EvalParams.corpus_size
            real_or_fake = "fake"
        else:
            subparts = ["all"]
            corpus_size = "no_limit"
            real_or_fake = "real"
        return subparts, corpus_size, real_or_fake

    @classmethod
    def build_corpora(cls, parallel: bool = False, data_sets: List[str] = None, filters: List[str] = None):
        if data_sets is None:
            data_sets = EvalParams.data_sets
        if filters is None:
            filters = EvalParams.filters

        data_set_bar = tqdm(data_sets, total=len(data_sets), desc="2 Operate on dataset")
        for data_set in data_set_bar:
            data_set_bar.set_description(f'2 Operate on dataset >{data_set}<')
            data_set_bar.refresh()

            data_set = data_set.replace("_short", "")
            data_set = data_set.replace("_medium", "")
            data_set = data_set.replace("_large", "")
            subparts, corpus_size, real_or_fake = EvaluationUtils.attributes_based_on_data(data_set)

            subparts_bar = tqdm(subparts,
                                desc='1 Iterating through subpart')

            for number_of_subparts in subparts_bar:
                subparts_bar.set_description(f'1 Iterating through subpart >{number_of_subparts}<')
                subparts_bar.refresh()

                corpus = chunk_documents(data_set, number_of_subparts, EvalParams.corpus_size)

                EvaluationUtils.filter_parsing_loop(parallel, corpus, data_set, number_of_subparts,
                                                    corpus_size, real_or_fake, filters=filters)

    @classmethod
    def train_vecs(cls, parallel: bool = False, data_sets: List[str] = None, filters: List[str] = None,
                   vectorization_algorithms: List[str] = None):
        if data_sets is None:
            data_sets = EvalParams.data_sets
        if filters is None:
            filters = EvalParams.filters
        if vectorization_algorithms is None:
            vectorization_algorithms = EvalParams.vectorization_algorithms

        data_set_bar = tqdm(data_sets, total=len(data_sets), desc="1 Operate on dataset")
        for data_set in data_set_bar:
            data_set_bar.set_description(f'1 Operate on dataset >{data_set}<')
            data_set_bar.refresh()

            data_set = data_set.replace("_short", "")
            data_set = data_set.replace("_medium", "")
            data_set = data_set.replace("_large", "")

            subparts, corpus_size, real_or_fake = EvaluationUtils.attributes_based_on_data(data_set)

            subparts_bar = tqdm(subparts,
                                desc='2 Iterating through subpart')
            for nr_of_subparts in subparts_bar:
                subparts_bar.set_description(f'2 Iterating through subpart >{nr_of_subparts}<')
                subparts_bar.refresh()

                filter_bar = tqdm(filters, total=len(filters), desc="3 Apply Filter")
                for filter_mode in filter_bar:
                    filter_bar.set_description(f'3 Apply Filter >{filter_mode}<')
                    filter_bar.refresh()
                    corpus = Corpus.fast_load(nr_of_subparts,
                                              corpus_size,
                                              data_set,
                                              filter_mode,
                                              real_or_fake,
                                              load_entities=False,)
                    EvaluationUtils.vectorization_loop(parallel, corpus, data_set, filter_mode,
                                                       nr_of_subparts, corpus_size, real_or_fake,
                                                       vectorization_algorithms=vectorization_algorithms)

    @classmethod
    def run_evaluation(cls, parallel: bool = False, data_sets: List[str] = None, filters: List[str] = None,
                       vectorization_algorithms: List[str] = None, task_names: List[str] = None, result_dir: str = None):
        if data_sets is None:
            data_sets = EvalParams.data_sets
        if filters is None:
            filters = EvalParams.filters
        if vectorization_algorithms is None:
            vectorization_algorithms = EvalParams.vectorization_algorithms
        if task_names is None:
            task_names = EvalParams.task_names
        if result_dir is None:
            result_dir = "results"
        tuples = []

        experiment_table_name = "series_experiment_table"
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        final_path = os.path.join(result_dir, f"simple_{experiment_table_name}.csv")
        cache_path = os.path.join(result_dir, f"cache_{experiment_table_name}.csv")
        paper_path = os.path.join(result_dir, f"{experiment_table_name}.csv")

        res = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict())))

        for data_set in tqdm(data_sets, total=len(data_sets), desc=f"Evaluate datasets"):
            subparts, corpus_size, real_or_fake = EvaluationUtils.attributes_based_on_data(data_set)
            if data_set.endswith('_short') or data_set.endswith('_medium') or data_set.endswith('_large'):
                splitted = data_set.split('_')
                size = splitted[-1]
                data_set = '_'.join(splitted[:-1])
            else:
                size = ""
            for number_of_subparts in subparts:
                for filter_mode in tqdm(filters, total=len(filters), desc=f"Evaluate filters"):

                    corpus = Corpus.fast_load(number_of_subparts,
                                              corpus_size,
                                              data_set,
                                              filter_mode,
                                              real_or_fake,
                                              load_entities=False)
                    if size:
                        corpus = corpus.length_sub_corpora_of_size(size)
                    vec_bar = tqdm(vectorization_algorithms,
                                   total=len(vectorization_algorithms),
                                   desc=f"Evaluate algorithm")

                    if parallel:
                        tuple_list_results = Parallel(n_jobs=EvalParams.num_cores)(
                            delayed(cls.eval_vec_loop_eff)(corpus,
                                                           number_of_subparts,
                                                           EvalParams.corpus_size,
                                                           data_set,
                                                           size,
                                                           filter_mode,
                                                           vectorization_algorithm,
                                                           real_or_fake,
                                                           task_names=task_names,
                                                           df_path=final_path)
                            for vectorization_algorithm in vec_bar)
                    else:
                        tuple_list_results = [cls.eval_vec_loop_eff(corpus,
                                                                    number_of_subparts,
                                                                    EvalParams.corpus_size,
                                                                    data_set,
                                                                    size,
                                                                    filter_mode,
                                                                    vectorization_algorithm,
                                                                    real_or_fake,
                                                                    task_names=task_names,
                                                                    df_path=final_path)
                                              for vectorization_algorithm in vec_bar]

                    for subpart_nr, data, filt_mod, vec_algo, results in tuple_list_results:
                        # results = results[results != np.array(None)]
                        res[subpart_nr][data][filt_mod][vec_algo] = results
        writing_mode = "w"
        header = True
        for data_set in tqdm(data_sets, total=len(data_sets),
                             desc="Store final results for dataset"):
            subparts, _, _ = EvaluationUtils.attributes_based_on_data(data_set)
            for number_of_subparts in subparts:
                for filter_mode in tqdm(filters, total=len(filters),
                                        desc="Store final results for filter"):
                    # print(data_set, res["all"].keys())
                    list_results = res[number_of_subparts][data_set][filter_mode]

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
                                tuples.extend(cls.aggregate_results(number_of_subparts, data_set, task_name, metric,
                                                                    filter_mode, metric_results,
                                                                    cache_path,
                                                                    vectorization_algorithms=vectorization_algorithms))
                        # tuples.extend(metric_tuples)

                    else:

                        tuples.extend(cls.aggregate_results(number_of_subparts, data_set, "Series",
                                                            EvalParams.evaluation_metric.__name__,
                                                            filter_mode, list_results, cache_path,
                                                            vectorization_algorithms=vectorization_algorithms))
        tuples = [tup for tup in tuples if tup != "None"]
        print(tuples)
        if len(tuples) > 0:
            df = pd.DataFrame(tuples, columns=['Series_length', 'Dataset', 'Task', 'Metric', 'Algorithm',
                                               'Filter', 'Score', 'Median'])
            print(df)
            df.to_csv(final_path, index=False, mode=writing_mode, header=header)
            print(EvaluationUtils.build_paper_table(df, paper_path))
            # print(EvaluationUtils.create_paper_table(df, paper_path, metrics=["prec", "prec01",
            #                                                                   "prec03", "prec05", "prec10"]))

    @classmethod
    def already_computed(cls, df_path, number_of_subparts, data_set, data_set_size, filter_mode,
                         vectorization_algorithm, tasks):
        if df_path:
            df = pd.read_csv(df_path)
            if data_set_size:
                data_set = f'{data_set}_{data_set_size}'
            len_count = 0
            for task in tasks:
                # print(number_of_subparts, data_set, data_set_size, filter_mode,
                #          vectorization_algorithm, task)
                # print(df)
                # print(df.loc[(df['Series_length'] == number_of_subparts)])
                # print(df.loc[(df['Dataset'] == data_set)])
                # print(df.loc[(df['Filter'] == filter_mode)])
                # print(df.loc[(df['Algorithm'] == vectorization_algorithm)])
                # print(df.loc[(df['Task'] == task)])
                # print(task, str(task))
                # print(task,  df['Task']==str(task))
                filtered = df.loc[(df['Series_length'] == number_of_subparts) &
                                  (df['Dataset'] == data_set) &
                                  (df['Filter'] == filter_mode) &
                                  (df['Algorithm'] == vectorization_algorithm) &
                                  (df['Task'] == str(task))]
                # print(filtered)
                if len(filtered) > 0:
                    len_count += 1
            # print(len_count)
            if len_count == len(tasks):
                return number_of_subparts, data_set, filter_mode, vectorization_algorithm, "TEMP"
        return None

    @classmethod
    def eval_vec_loop_eff(cls, corpus: Corpus, number_of_subparts, corpus_size, data_set, data_set_size,
                          filter_mode, vectorization_algorithm, real_or_fake: str,
                          task_names: List[str] = None, df_path: str = None):
        topn = 100
        summation_method = "NF"
        if task_names is None:
            task_names = EvalParams.task_names

        if corpus.series_dict:
            reverted = Utils.revert_dictionaried_list(corpus.series_dict)
        else:
            reverted = None

        tasks = [EvaluationTask.create_from_name(task_name, reverted=reverted, corpus=corpus, topn=topn)
                 for task_name in task_names]

        # log_result = cls.already_computed(df_path, number_of_subparts, data_set, data_set_size,
        #                                   filter_mode, vectorization_algorithm, tasks)
        # if log_result:
        #     return log_result

        # print('at', vec_path, real_or_fake)

        # print(vectorization_algorithm)
        wmd_sims = None
        if "wmd".lower() == vectorization_algorithm or "WordMoversDistance".lower() == vectorization_algorithm.lower():
            # wmd = WordMoversDistance(corpus=corpus, embedding_path='E:/embeddings/glove.6B.100d.txt', top_n_docs=topn,
            #                          top_n_words=100)
            WordMoversDistance.embedding_path = 'E:/embeddings/glove.6B.300d.txt'
            topn_words = 100
            wmd_sims = WordMoversDistance.similarities(path=f"D:/models/wmd/{data_set}_{topn}d_{topn_words}w.json",
                                                       corpus=corpus,
                                                       top_n_docs=topn,
                                                       top_n_words=topn_words)
            doctags = corpus.documents.keys()

            series = False
            vectors = None
        else:
            vec_path = Vectorization.build_vec_file_name(number_of_subparts,
                                                         corpus_size,
                                                         data_set,
                                                         filter_mode,
                                                         vectorization_algorithm,
                                                         real_or_fake,
                                                         allow_combination=True)
            vectors, summation_method = Vectorization.my_load_doc2vec_format(vec_path)
            # try:
            #     vectors = Vectorization.my_load_doc2vec_format(vec_path)
            # except FileNotFoundError:
            #     if "_o_" in vectorization_algorithm:
            #         vec_splitted = vectorization_algorithm.split("_o_")
            #         base_algorithm = vec_splitted[0]
            #         focus_facette = vec_splitted[1]
            #
            #         vec_path = Vectorization.build_vec_file_name(number_of_subparts,
            #                                                      corpus_size,
            #                                                      data_set,
            #                                                      filter_mode,
            #                                                      base_algorithm,
            #                                                      real_or_fake,
            #                                                      allow_combination=True)
            #         vectors = Vectorization.my_load_doc2vec_format(vec_path)
            #         summation_method = focus_facette
            #     else:
            #         raise FileNotFoundError
            doctags = vectors.docvecs.doctags.keys()
            # print(len(doctags))
            # print(doctags)
            doctags = [doctag for doctag in doctags if corpus.vector_doc_id_base_in_corpus(doctag)]
            # print(len(doctags))
            # print(doctags)
            # data_set = data_set + "_series"

            if "series" not in data_set and "dta" not in data_set:
                doctags = [doctag for doctag in doctags if doctag[-1].isdigit() and Vectorization.doctag_filter(doctag)]
                series = False
            else:
                doctags = [doctag for doctag in doctags if doctag[-1].isdigit()]
                series = True
            # print(series)
        # print('>', len(doctags))
        results = []
        task_results = defaultdict(list)
        # test_results = []
        topn_value = 1
        if EvalParams.ignore_same:
            topn_value = 1


        # print(doctags)
        all_doc_id_dict = defaultdict(dict)
        missed_dict = defaultdict(dict)
        for doc_id in doctags:
            # print(doc_id)
            # topn = len(corpus.series_dict[reverted[doc_id]])

            if topn > topn_value:
                # print('#############', doc_id, len(doctags))
                if wmd_sims:
                    sim_documents = wmd_sims[doc_id]
                else:
                    sim_documents = Vectorization.most_similar_documents(vectors, corpus,
                                                                         positives=[doc_id],
                                                                         feature_to_use=summation_method,
                                                                         topn=topn,
                                                                         print_results=False,
                                                                         series=series)

                # sim_documents = [(sim_document[0], sim_document[1]) for sim_document in sim_documents
                #                  if doctag_filter(sim_document[0])]
                # print('sim', len(sim_documents))
                for task in tasks:
                    # print(task.nr_of_possible_matches(doc_id), task.__class__)
                    # print(doc_id, sim_documents)
                    if isinstance(task, SeriesTask):
                        if doc_id not in task.reverted:
                            # print(doc_id)
                            continue
                    try:
                        metric_results, doc_id_dict, missed = EvalParams.evaluation_metric(sim_documents, doc_id,
                                                                                           task,
                                                                                           ignore_same=EvalParams.ignore_same)
                        all_doc_id_dict[str(task)].update(doc_id_dict)
                        missed_dict[str(task)].update(missed)
                        task_results[str(task)].append(metric_results)
                        results.append(metric_results)
                    except KeyError:
                        pass

        # task_dict = {}
        # for task in tasks:
        #     print(str(task), task.correct)
        #     print(str(task), task.uncorrect)
        #     task_dict[str(task)] = {"correct": task.correct, "uncorrect": task.uncorrect, "ground_truth": task.truth}
        # with open(f'results/logged_decisions/{number_of_subparts}_{corpus_size}_{data_set}_{data_set_size}'
        #           f'_{filter_mode}_{vectorization_algorithm}_{real_or_fake}_res.json', 'w') as fp:
        #     json.dump(task_dict, fp, indent=1)

        log = []
        for task, doc_id_dict in all_doc_id_dict.items():
            for doc_id, sim_docs in doc_id_dict.items():
                for sim_doc in sim_docs:
                    log.append((vectorization_algorithm, task, doc_id, corpus.documents[doc_id], sim_doc[0],
                                corpus.documents[sim_doc[0]], sim_doc[1]))
        log_df = pd.DataFrame(log, columns=["Algorithm", "Task", "doc_id", "Doc", "sim_doc_id", "Sim Doc", "Correct"])
        log_df.to_csv(f'results/logged_decisions/{vectorization_algorithm}_neighbors.csv')
        missed_log = []
        for task, doc_id_dict in missed_dict.items():
            for doc_id, sim_docs in doc_id_dict.items():
                for sim_doc in sim_docs:
                    # print(sim_doc)
                    missed_doc = sim_doc
                    if sim_doc in corpus.documents:
                        missed_doc = corpus.documents[sim_doc]
                    missed_log.append((vectorization_algorithm, task, doc_id, corpus.documents[doc_id], sim_doc, missed_doc
                                       ))
        log_df = pd.DataFrame(missed_log, columns=["Algorithm", "Task", "doc_id", "Doc", "missed doc_id", "Missed Doc"])
        log_df.to_csv(f'results/logged_decisions/{vectorization_algorithm}_missed.csv')

        # print('res', len(results))
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

        if data_set_size:
            data_set = f'{data_set}_{data_set_size}'
        # data_set = data_set.replace('_series', '')
        return number_of_subparts, data_set, filter_mode, vectorization_algorithm, final_task_results

    @classmethod
    def aggregate_results(cls, subpart_nr: Union[str, int], data_set: str, task_name: str, metric_name: str,
                          filter_mode: str,
                          results_as_dict: Dict[str, np.ndarray],
                          cache_path: str,
                          vectorization_algorithms: List[str] = None):
        if vectorization_algorithms is None:
            vectorization_algorithms = EvalParams.vectorization_algorithms
        tuples = []

        results_as_dict = {key: [result for result in results if result is not None]
                           for key, results in results_as_dict.items()}

        # significance_dict = EvaluationMath.one_way_anova(results_as_dict)
        vec_bar = tqdm(vectorization_algorithms, total=len(vectorization_algorithms),
                       desc="Store final results for algorithm")
        # Scoring

        for vectorization_algorithm in vec_bar:
            results = results_as_dict[vectorization_algorithm]
            # print(results)
            if results == list("TEMP"):
                tuples.append("None")
                continue
            # print('>|', results)
            # results = [res for res in results if res is not None]
            # print('>|', results)
            # sig = significance_dict[vectorization_algorithm]
            score, deviation = EvaluationMath.mean(results)
            # vectorization_results[vectorization_algorithm] = score, deviation
            observation = (subpart_nr, data_set, task_name, metric_name, vectorization_algorithm, filter_mode,
                           f'{score:.4f} ± {deviation:.4f}',
                           EvaluationMath.median(results))
            # observation = (subpart_nr, data_set, task_name, metric_name, vectorization_algorithm, filter_mode,
            #                f'{score:.4f} ± {deviation:.4f} [{sig}]',
            #                EvaluationMath.median(results))
            tuples.append(observation)

            df_obs = pd.DataFrame([observation],
                                  columns=['Series_length', 'Dataset', 'Task', 'Metric', 'Algorithm',
                                           'Filter', 'Score', 'Median'])
            df_obs.to_csv(cache_path, mode='a', header=(not os.path.exists(cache_path)), index=False)
        return tuples

    @classmethod
    def filter_parsing_loop(cls, parallel: bool, corpus: Corpus, data_set: str, number_of_subparts: Union[str, int],
                            corpus_size: str, real_or_fake: str, filters: List[str] = None):
        if filters is None:
            filters = EvalParams.filters
        filter_bar = tqdm(filters, total=len(filters), desc="3 Calculate filter_mode results")
        if parallel:
            Parallel(n_jobs=EvalParams.num_cores)(
                delayed(EvaluationUtils.store_corpus_to_parameters_eff)(corpus,
                                                                        number_of_subparts,
                                                                        corpus_size,
                                                                        data_set,
                                                                        filter_mode,
                                                                        real_or_fake)
                for filter_mode in filter_bar)
        else:
            [EvaluationUtils.store_corpus_to_parameters_eff(corpus,
                                                            number_of_subparts,
                                                            corpus_size,
                                                            data_set,
                                                            filter_mode,
                                                            real_or_fake)
             for filter_mode in filter_bar]

    @classmethod
    def vectorization_loop(cls, parallel: bool, corpus: Corpus, data_set: str, filter_mode: str,
                           number_of_subparts: Union[str, int], corpus_size: str, real_or_fake: str,
                           vectorization_algorithms: List[str] = None):
        if vectorization_algorithms is None:
            vectorization_algorithms = EvalParams.vectorization_algorithms

        vec_bar = tqdm(vectorization_algorithms, total=len(vectorization_algorithms),
                       desc="4 Vectorize")
        if parallel:
            Parallel(n_jobs=EvalParams.num_cores)(delayed(EvaluationUtils.sep_vec_calc_eff)(corpus,
                                                                                            number_of_subparts,
                                                                                            corpus_size,
                                                                                            data_set,
                                                                                            filter_mode,
                                                                                            vec_algorithm,
                                                                                            real_or_fake)
                                                  for vec_algorithm in vec_bar)
        else:
            [EvaluationUtils.sep_vec_calc_eff(corpus,
                                              number_of_subparts,
                                              corpus_size,
                                              data_set,
                                              filter_mode,
                                              vectorization_algorithm,
                                              real_or_fake)
             for vectorization_algorithm in vec_bar]

    @staticmethod
    def sep_vec_calc_eff(corpus, number_of_subparts, corpus_size, data_set, filter_mode,
                         vectorization_algorithm, fake):
        print(vectorization_algorithm)
        if vectorization_algorithm.lower() == "wmd" or vectorization_algorithm.lower() == "wordmoversdistance":
            return
        vec_file_name = Vectorization.build_vec_file_name(number_of_subparts,
                                                          corpus_size,
                                                          data_set,
                                                          filter_mode,
                                                          vectorization_algorithm,
                                                          fake)
        # print(vec_file_name)
        if not os.path.isfile(vec_file_name):
            EvaluationUtils.store_vectors_to_parameters(corpus,
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
            corpus.filter_on_copy_mem_eff(filtered_corpus_dir=filtered_corpus_dir, mode=filter_mode)
            # corpus.save_corpus_adv(filtered_corpus_dir)

    @staticmethod
    def store_vectors_to_parameters(corpus, number_of_subparts, corpus_size, data_set, filter_mode,
                                    vectorization_algorithm, fake):
        vec_file_name = Vectorization.build_vec_file_name(number_of_subparts,
                                                          corpus_size,
                                                          data_set,
                                                          filter_mode,
                                                          vectorization_algorithm,
                                                          fake)
        if not os.path.isfile(vec_file_name):
            Vectorizer.algorithm(input_str=vectorization_algorithm,
                                 corpus=corpus,
                                 save_path=vec_file_name,
                                 return_vecs=False)
        # else:
        #     logging.info(f'{vec_file_name} already exists, skip')


class EvalParams:
    config = ConfigLoader.get_config()
    min_number_of_subparts = 2
    max_number_of_subparts = 3
    corpus_size = "no_limit"
    num_cores = int(0.75 * multiprocessing.cpu_count())

    ignore_same = True
    # evaluation_metric = EvaluationMetric.ap
    # evaluation_metric = EvaluationMetric.precision
    # evaluation_metric = EvaluationMetric.ndcg
    evaluation_metric = EvaluationMetric.multi_metric

    data_sets = [
        # "classic_gutenberg_fake_series",
        # "german_series",
        # "classic_gutenberg",
        # # "german_series_short",
        # # "german_series_medium",
        # # "german_series_large",
        "goodreads_genres",
        # "dta",
        # "dta_series",
        # "summaries",

        # "goodreads_genres_short",
        # "goodreads_genres_medium",
        # "goodreads_genres_large",
        # "tagged_german_books",
        # "german_books",
        # "litrec",
    ]
    filters = [
        # "no_filter",
        # "specific_words_moderate",
        "specific_words_strict",
        # "named_entities",
        # "common_words_strict",
        # "common_words_strict_general_words_sensitive",
        # "common_words_relaxed",
        # "common_words_relaxed_general_words_sensitive",
        # "common_words_doc_freq",
        # "stopwords",
        # "nouns",
        # "verbs",
        # "adjectives",
        # "avn"
    ]
    vectorization_algorithms = [
        # "wmd",
        # "bow",
        # "avg_wv2doc",
        # "avg_wv2doc_restrict10000",
        # "doc2vec",
        # "doc2vec_chunk",
        # "psif",
        # "book2vec",
        # "bert_pt",
        # "bert_pt_chunk",
        "bert_sentence_based_1000_pt",
        # "roberta_pt",
        # "roberta_pt_chunk",
        "roberta_sentence_based_1000_pt",
        # "xlm_pt",
        # "xlm_pt_chunk",
        "xlm_sentence_based_1000_pt",
        # "bert_sentence_based_1000_pt",
        # "longformer",
        # "roberta_pt",
        # "xlm_pt",
        # "roberta_sentence_based_1000_pt",
        # "xlm_sentence_based_1000_pt",

        # "doc2vec_dbow",
        # "book2vec",
        # "book2vec_dbow",
        # "avg_wv2doc_pretrained",
        # "doc2vec_pretrained",
        # ,
        # # "longformer_untuned"

        # "book2vec_wo_raw",
        # "book2vec_dbow_wo_raw",
        # "book2vec_dbow_wo_raw",

        # "book2vec_pretrained",
        # "book2vec_avg",
        # "book2vec_auto",
        # "book2vec_concat",
        # "book2vec_pca",
        # "doc2vec_dim50",
        # "doc2vec_dim100",
        # "doc2vec_dim300",
        # "doc2vec_dim500",
        # "doc2vec_dim700",
        # "doc2vec_dim900",

        # "book2vec_dim50",
        # "book2vec_dim50_concat",
        # "book2vec_dim100",
        # "book2vec_dim100_concat",
        # "book2vec_dim300",
        # "book2vec_dim300_concat",
        # "book2vec_dim500",
        # "book2vec_dim500_concat",
        # "book2vec_dim700",
        # "book2vec_dim700_concat",
        # "book2vec_dim900",
        # "book2vec_dim900_concat",




        # "topic2vec",
        # "book2vec_window",
        # "book2vec_o_raw",
        # "book2vec_o_loc",
        # "book2vec_o_time",
        # "book2vec_o_sty",
        # "book2vec_o_atm",

        # "book2vec_chunk",
        # "book2vec_chunk_window",
        # "book2vec_chunk_o_raw",
        # "book2vec_chunk_o_loc",
        # "book2vec_chunk_o_time",
        # "book2vec_chunk_o_sty",
        # "book2vec_chunk_o_atm",

        # "book2vec_chunk_facet",
        # "book2vec_chunk_facet_window",
        # "book2vec_chunk_facet_o_raw",
        # "book2vec_chunk_facet_o_loc",
        # "book2vec_chunk_facet_o_time",
        # "book2vec_chunk_facet_o_sty",
        # "book2vec_chunk_facet_o_atm",

        # "book2vec_wo_raw",
        # "book2vec_wo_loc",
        # "book2vec_wo_time",
        # "book2vec_wo_sty",
        # "book2vec_wo_atm",
        # "book2vec_w2v",

        # "book2vec_adv",
        # "book2vec_adv_avg",
        # "book2vec_adv_concat",
        # "book2vec_adv_pca",
        # "book2vec_adv_auto",

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
        # 'book2vec_bert',
        # 'bert'
    ]

    task_names = [
        # "SeriesTask",
        "AuthorTask",
        "GenreTask"
    ]

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
    # EvalParams
    # EvaluationUtils.build_corpora()
    EvaluationUtils.train_vecs()
    EvaluationUtils.run_evaluation()
    # print(EvaluationUtils.create_paper_table("results/simple_series_experiment_table.csv", "results/z_table_gb.csv",
    #                                          used_metrics=["ndcg", "prec", "prec01", "prec03", "prec05", "prec10",
    #                                                        "length_metric"],
    #                                          filters=["common_words_relaxed", "common_words_strict",
    #                                                   "common_words_strict_general_words_sensitive",
    #                                                   "common_words_relaxed_general_words_sensitive",
    #                                                   "common_words_doc_freq"]))

    print(EvaluationUtils.create_paper_table("results/simple_series_experiment_table.csv", "results/z_table.csv",
                                             used_metrics=["ndcg", "f_prec", "f_prec01", "f_prec03", "f_prec05",
                                                           "f_prec10",
                                                           "length_metric"],
                                             filters=["no_filter",
                                                      "specific_words_moderate",
                                                      "specific_words_strict"
                                                      ]))
    # print(EvaluationUtils.create_paper_table("results/simple_series_experiment_table.csv", "results/z_table_gb.csv",
    #                                          used_metrics=["ndcg", "prec", "prec01", "prec03", "prec05",
    #                                                        "prec10",
    #                                                        "length_metric"],
    #                                          filters=["no_filter"]))
    # print(EvaluationUtils.create_paper_table("results/simple_series_experiment_table.csv", "results/z_table_gb.csv",
    #                                          used_metrics=["ndcg", "rec", "pec01", "rec03", "rec05",
    #                                                        "rec10",
    #                                                        "length_metric"],
    #                                          filters=["no_filter",
    #                                                   "specific_words_moderate",
    #                                                   "specific_words_strict"
    #                                                   ]))

    # print(EvaluationUtils.create_paper_table("results/simple_series_experiment_table.csv", "results/z_table_gb.csv",
    #                                          used_metrics=["ndcg", "f1", "f101", "f103", "f105",
    #                                                        "f110",
    #                                                        "length_metric"],
    #                                          filters=["no_filter"]))
