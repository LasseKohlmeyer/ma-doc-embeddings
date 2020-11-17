import logging
import logging.config
import multiprocessing
import os
from collections import defaultdict, namedtuple
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


class EvaluationMetric:
    @staticmethod
    def precision_scores(sim_documents, doc_id: str, corpus: Corpus, reverted):
        hard_correct = 0
        # soft_correct = 0
        # print(reverted)
        for sim_doc_id, _ in sim_documents:
            if reverted[doc_id] == reverted[sim_doc_id]:
                hard_correct += 1
            # if corpus.documents[doc_id].authors == corpus.documents[sim_doc_id].authors:
            #     soft_correct += 1

        hard_correct = hard_correct / len(sim_documents)
        # soft_correct = soft_correct / len(sim_documents)
        return hard_correct

    @staticmethod
    def ap(sim_documents, doc_id: str, corpus: Corpus, reverted):
        # print(reverted)
        k = 1
        prec_values_at_k = []
        correct_ones = []
        for sim_doc_id, _ in sim_documents:
            if reverted[doc_id] == reverted[sim_doc_id]:
                correct_ones.append(k)
                prec_values_at_k.append(len(correct_ones)/k)

            k += 1
        if len(prec_values_at_k) > 0:
            ap = sum(prec_values_at_k)/len(prec_values_at_k)
        else:
            ap = 0
        return ap

    @staticmethod
    def ndcg_c(sim_documents, doc_id: str, corpus: Corpus, reverted):
        # print(reverted)
        ground_truth_values = []
        predicted_values = []
        for sim_doc_id, sim in sim_documents:
            predicted_values.append(sim)

            if reverted[doc_id] == reverted[sim_doc_id]:
                ground_truth_values.append(1)
            else:
                ground_truth_values.append(0)

        ndcg = metrics.ndcg_score(np.array([ground_truth_values]),
                                  np.array([predicted_values]))
        return ndcg

    @staticmethod
    def ndcg_b(sim_documents, doc_id: str, corpus: Corpus, reverted):
        # print(reverted)
        ground_truth_values = []
        predicted_values = []
        c = 0
        for sim_doc_id, _ in sim_documents:
            if reverted[doc_id] == reverted[sim_doc_id]:
                predicted_values.append(1)
            else:
                predicted_values.append(0)

            if c < len(corpus.series_dict[doc_id]):
                ground_truth_values.append(1)
            else:
                ground_truth_values.append(0)
            c += 1

        ndcg = metrics.ndcg_score(np.array([ground_truth_values]),
                                  np.array([predicted_values]))
        return ndcg

    @staticmethod
    def ndcg_a(sim_documents, doc_id: str, corpus: Corpus, reverted):
        # print(reverted)
        ground_truth_values = []
        predicted_values = []

        for sim_doc_id, _ in sim_documents:
            predicted_values.append(1)
            if reverted[doc_id] == reverted[sim_doc_id]:
                ground_truth_values.append(1)
            else:
                ground_truth_values.append(0)

        ndcg = metrics.ndcg_score(np.array([ground_truth_values]),
                                  np.array([predicted_values]))
        return ndcg


class Evaluation:
    evaluation_metric = EvaluationMetric.precision_scores

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
        ap_values = []
        for doc_id in sample:
            sim_documents = Vectorizer.most_similar_documents(vectors, corpus,
                                                              positives=[doc_id],
                                                              feature_to_use="NF",
                                                              topn=topn,
                                                              print_results=False)
            hard_correct = cls.evaluation_metric(sim_documents, doc_id, corpus, reverted)
            results.append(hard_correct)
            ap_values.append(EvaluationMetric.ap(sim_documents, doc_id, corpus, reverted))

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
        df_table = cache_df
        df_table["Dataset"] = pd.Categorical(df_table["Dataset"],
                                             categories=df_table["Dataset"].unique(), ordered=True)
        df_table["Algorithm"] = pd.Categorical(df_table["Algorithm"],
                                               categories=df_table["Algorithm"].unique(), ordered=True)
        df_table = df_table.set_index(['Series_length', 'Dataset', 'Algorithm'])
        df_table["Filter"] = pd.Categorical(df_table["Filter"], categories=df_table["Filter"].unique(), ordered=True)
        df_table = df_table.pivot(columns='Filter')['Score']
        df_table.to_csv(out_path, index=True, encoding="utf-8")
        return df_table


class EvaluationRun:
    config = ConfigLoader.get_config()
    min_number_of_subparts = 10
    max_number_of_subparts = 10
    corpus_size = 201
    num_cores = int(0.75*multiprocessing.cpu_count())

    data_sets = [
        # "summaries",
        # "tagged_german_books"
        "german_books",
        # "german_series"
        # "litrec",

    ]
    filters = [
        "no_filter",
        "named_entities",
        "common_words",
        "stopwords",
        # "nouns",
        # "verbs",
        # "adjectives",
        "avn"
    ]
    vectorization_algorithms = [
        # "avg_wv2doc",
        # "doc2vec",
        # "book2vec",
        # "book2vec_wo_raw",
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
        filtered_corpus_file_name = Corpus.build_corpus_file_name(number_of_subparts,
                                                                  corpus_size,
                                                                  data_set,
                                                                  filter_mode,
                                                                  fake)
        if not os.path.isfile(filtered_corpus_file_name):
            common_words_dict = corpus.get_common_words(corpus.series_dict)
            corpus = corpus.filter_on_copy(mode=filter_mode, common_words=common_words_dict)
            corpus.save_corpus(filtered_corpus_file_name)

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
                                                     f'{data_set}_{cls.corpus_size}.json')
                annotated_series_corpus_path = os.path.join(cls.config["system_storage"]["corpora"],
                                                            f'{data_set}_{number_of_subparts}_'
                                                            f'{cls.corpus_size}_series.json')
                # Corpus +Document
                try:
                    # check if series corpus exists
                    corpus = Corpus(annotated_series_corpus_path)
                except FileNotFoundError:
                    try:
                        # check if general corpus exists
                        corpus = Corpus(annotated_corpus_path)
                        corpus = Preprocesser.filter_too_small_docs_from_corpus(corpus)
                        corpus, _ = corpus.fake_series(number_of_sub_parts=number_of_subparts)
                        corpus.save_corpus(annotated_series_corpus_path)
                    except FileNotFoundError:
                        # load from raw data
                        corpus = DataHandler.load_corpus(data_set)
                        if cls.corpus_size == "no_limit":
                            corpus = Preprocesser.annotate_corpus(corpus)
                        else:
                            corpus = corpus.sample(cls.corpus_size, seed=42)
                            corpus = Preprocesser.annotate_corpus(corpus)
                            # corpus = Preprocesser.annotate_corpus(corpus[:cls.corpus_size])
                        corpus.save_corpus(annotated_corpus_path)
                        corpus = Preprocesser.filter_too_small_docs_from_corpus(corpus)
                        corpus, _ = corpus.fake_series(number_of_sub_parts=number_of_subparts)
                        corpus.save_corpus(annotated_series_corpus_path)
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
                    filtered_corpus_file_name = Corpus.build_corpus_file_name(number_of_subparts,
                                                                              cls.corpus_size,
                                                                              data_set,
                                                                              filter_mode,
                                                                              "fake")
                    corpus = Corpus(filtered_corpus_file_name)

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
    def run_evaluation(cls, parallel: bool = True):
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
                    corpus_path = Corpus.build_corpus_file_name(number_of_subparts,
                                                                cls.corpus_size,
                                                                data_set,
                                                                filter_mode,
                                                                "fake")
                    corpus = Corpus(corpus_path)
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

    evaluation_metric = EvaluationMetric.precision_scores
    data_sets = [
        "german_series"
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

    @classmethod
    def build_real_series_corpora(cls, parallel: bool = False):
        data_set_bar = tqdm(cls.data_sets, total=len(cls.data_sets), desc="2 Operate on dataset")
        for data_set in data_set_bar:
            data_set_bar.set_description(f'2 Operate on dataset >{data_set}<')
            data_set_bar.refresh()
            annotated_corpus_path = os.path.join(cls.config["system_storage"]["corpora"], f'{data_set}.json')
            try:
                # check if general corpus exists
                corpus = Corpus(annotated_corpus_path)
            except FileNotFoundError:
                # load from raw data
                corpus = DataHandler.load_corpus(data_set)
                corpus = Preprocesser.annotate_corpus(corpus)
                corpus.save_corpus(annotated_corpus_path)

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
                filtered_corpus_file_name = Corpus.build_corpus_file_name("all",
                                                                          "no_limit",
                                                                          data_set,
                                                                          filter_mode,
                                                                          "real")
                corpus = Corpus(filtered_corpus_file_name)

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
    def run_evaluation(cls, parallel: bool = False):
        tuples = []
        result_dir = "results"
        experiment_table_name = "series_experiment_table"
        final_path = os.path.join(result_dir, f"simple_{experiment_table_name}.csv")
        cache_path = os.path.join(result_dir, f"cache_{experiment_table_name}.csv")
        paper_path = os.path.join(result_dir, f"{experiment_table_name}.csv")

        ParameterConfig = namedtuple('ParameterConfig', ['number_of_subparts', 'data_set', 'filter_mode',
                                                         'vectorization_algorithm'])
        parameter_tuples = []
        for data_set in cls.data_sets:
            for filter_mode in cls.filters:
                for vectorization_algorithm in cls.vectorization_algorithms:
                    parameter_tuples.append(ParameterConfig("all", data_set, filter_mode,
                                                            vectorization_algorithm))
        parameter_tuples_bar = tqdm(parameter_tuples, desc='Benchmarking vectors')
        if parallel:
            tuple_list_results = Parallel(n_jobs=cls.num_cores)(
                delayed(RealSeriesEvaluationRun.eval_vec_loop)(params.number_of_subparts,
                                                               "no_limit",
                                                               params.data_set,
                                                               params.filter_mode,
                                                               params.vectorization_algorithm,
                                                               "real")
                for params in parameter_tuples_bar)
        else:
            tuple_list_results = [RealSeriesEvaluationRun.eval_vec_loop(params.number_of_subparts,
                                                                        "no_limit",
                                                                        params.data_set,
                                                                        params.filter_mode,
                                                                        params.vectorization_algorithm,
                                                                        "real")
                                  for params in parameter_tuples_bar]

        res = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict())))
        for nr_of_subparts, data_set, filter_mode, vectorization_algorithm, results in tuple_list_results:
            res[nr_of_subparts][data_set][filter_mode][vectorization_algorithm] = results

        # print(res[10]['summaries']['common_words']['book2vec'])
        # print(res)

        for data_set in cls.data_sets:
            for filter_mode in cls.filters:
                list_results = res["all"][data_set][filter_mode]

                # Evaluation.t_test(list_results)
                significance_dict = EvaluationMath.one_way_anova(list_results)

                # Scoring
                for vectorization_algorithm in cls.vectorization_algorithms:
                    results = list_results[vectorization_algorithm]
                    sig = significance_dict[vectorization_algorithm]
                    score, deviation = EvaluationMath.mean(results)
                    # vectorization_results[vectorization_algorithm] = score, deviation
                    observation = ("all", data_set, vectorization_algorithm, filter_mode,
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

    @classmethod
    def run_evaluation_eff(cls, parallel: bool = True):
        tuples = []
        result_dir = "results"
        experiment_table_name = "series_experiment_table"
        final_path = os.path.join(result_dir, f"simple_{experiment_table_name}.csv")
        cache_path = os.path.join(result_dir, f"cache_{experiment_table_name}.csv")
        paper_path = os.path.join(result_dir, f"{experiment_table_name}.csv")

        res = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict())))
        for data_set in tqdm(cls.data_sets, total=len(cls.data_sets), desc=f"Evaluate datasets"):
            for filter_mode in tqdm(cls.filters, total=len(cls.filters), desc=f"Evaluate filters"):
                corpus_path = Corpus.build_corpus_file_name("all",
                                                            "no_limit",
                                                            data_set,
                                                            filter_mode,
                                                            "real")
                corpus = Corpus(corpus_path)
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

                # Evaluation.t_test(list_results)
                significance_dict = EvaluationMath.one_way_anova(list_results)
                vec_bar = tqdm(cls.vectorization_algorithms, total=len(cls.vectorization_algorithms),
                               desc="Store final results afor lgorithm")
                # Scoring
                for vectorization_algorithm in vec_bar:
                    results = list_results[vectorization_algorithm]
                    sig = significance_dict[vectorization_algorithm]
                    score, deviation = EvaluationMath.mean(results)
                    # vectorization_results[vectorization_algorithm] = score, deviation
                    observation = ("all", data_set, vectorization_algorithm, filter_mode,
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

    @classmethod
    def eval_vec_loop(cls, number_of_subparts, corpus_size, data_set, filter_mode, vectorization_algorithm, fake):
        corpus_path = Corpus.build_corpus_file_name(number_of_subparts,
                                                    corpus_size,
                                                    data_set,
                                                    filter_mode,
                                                    fake)
        vec_path = Vectorizer.build_vec_file_name(number_of_subparts,
                                                  corpus_size,
                                                  data_set,
                                                  filter_mode,
                                                  vectorization_algorithm,
                                                  fake)

        vectors = Vectorizer.my_load_doc2vec_format(vec_path)
        corpus = Corpus(corpus_path)
        print(corpus.series_dict)
        reverted = Utils.revert_dictionaried_list(corpus.series_dict)
        doctags = vectors.docvecs.doctags.keys()
        doctags = [doctag for doctag in doctags if doctag[-1].isdigit()]

        results = []
        ap_values = []
        for doc_id in doctags:
            topn = len(corpus.series_dict[reverted[doc_id]])
            sim_documents = Vectorizer.most_similar_documents(vectors, corpus,
                                                              positives=[doc_id],
                                                              feature_to_use="NF",
                                                              topn=topn,
                                                              print_results=False)
            hard_correct = cls.evaluation_metric(sim_documents, doc_id, corpus, reverted)
            ap_values.append(EvaluationMetric.ap(sim_documents, doc_id, corpus, reverted))
            results.append(hard_correct)

        # results_avg, _ = Evaluation.similar_docs_avg(vectors, corpus, reverted, doctags, topn)
        # print(seed, Evaluation.mean(results_avg))

        results = np.array(results)

        return number_of_subparts, data_set, filter_mode, vectorization_algorithm, results

    @classmethod
    def eval_vec_loop_eff(cls, corpus, number_of_subparts, corpus_size, data_set, filter_mode, vectorization_algorithm):
        vec_path = Vectorizer.build_vec_file_name(number_of_subparts,
                                                  corpus_size,
                                                  data_set,
                                                  filter_mode,
                                                  vectorization_algorithm,
                                                  "real")

        vectors = Vectorizer.my_load_doc2vec_format(vec_path)
        reverted = Utils.revert_dictionaried_list(corpus.series_dict)
        doctags = vectors.docvecs.doctags.keys()
        doctags = [doctag for doctag in doctags if doctag[-1].isdigit()]

        results = []
        ap_values = []
        for doc_id in doctags:
            topn = len(corpus.series_dict[reverted[doc_id]])
            sim_documents = Vectorizer.most_similar_documents(vectors, corpus,
                                                              positives=[doc_id],
                                                              feature_to_use="NF",
                                                              topn=topn,
                                                              print_results=False)
            # hard_correct = EvaluationMetric.precision_scores(sim_documents, doc_id, corpus, reverted)
            hard_correct = cls.evaluation_metric(sim_documents, doc_id, corpus, reverted)
            ap_values.append(EvaluationMetric.ap(sim_documents, doc_id, corpus, reverted))
            results.append(hard_correct)

        # results_avg, _ = Evaluation.similar_docs_avg(vectors, corpus, reverted, doctags, topn)
        # print(seed, Evaluation.mean(results_avg))

        results = np.array(results)

        return number_of_subparts, data_set, filter_mode, vectorization_algorithm, results
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
    # todo workaround memory error
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logging.confi

    # EvaluationRun.build_corpora()
    # EvaluationRun.train_vecs()
    EvaluationRun.run_evaluation()

    # RealSeriesEvaluationRun.build_real_series_corpora()
    # RealSeriesEvaluationRun.train_real_series_vecs()
    RealSeriesEvaluationRun.run_evaluation_eff()
