import os
from collections import defaultdict
import random
from typing import Dict, List

import pandas as pd
from scipy.stats import stats

from corpus_structure import Corpus, DataHandler, Preprocesser
from success_predict import mcnemar_sig_text, chi_square_test
from vectorization import Vectorizer
from vectorization_utils import Vectorization
import numpy as np


def get_percentage_of_correctly_labeled(vectors, human_assessment_df: pd.DataFrame, doc_id_mapping: Dict[str, str],
                                        facet_mapping: Dict[str, str], use_sum: bool):
    # reverted_facets = {value: key for key, value in facet_mapping.items()}
    correctly_assessed = []
    facet_wise = defaultdict(list)
    random_baseline = False
    skip_count = 0
    agreement_store = defaultdict(list)
    for i, row in human_assessment_df.iterrows():
        book1 = doc_id_mapping[row["Book 1"]]
        book2 = doc_id_mapping[row["Book 2"]]
        book3 = doc_id_mapping[row["Book 3"]]
        if use_sum:
            facet = facet_mapping["total"]
        else:
            facet = facet_mapping[row["Facet"]]
        selection = row["Selection"]
        if selection == "skip" or selection == "unsure":
            skip_count += 1
            continue

        if random_baseline:
            if int(row["Selected Answer Nr."]) == random.randint(1, 3):
                correctly_assessed.append(1)
                facet_wise[row["Facet"]].append(1)
            else:
                correctly_assessed.append(0)
                facet_wise[row["Facet"]].append(0)
        else:
            sim_1 = Vectorization.facet_sim(model_vectors=vectors, doc_id_a=book1, doc_id_b=book2, facet_name=facet)
            sim_2 = Vectorization.facet_sim(model_vectors=vectors, doc_id_a=book1, doc_id_b=book3, facet_name=facet)
            sim_3 = Vectorization.facet_sim(model_vectors=vectors, doc_id_a=book2, doc_id_b=book3, facet_name=facet)

            if int(row["Selected Answer Nr."]) == 1 and sim_1 > sim_2 and sim_1 > sim_3:
                correctly_assessed.append(1)
                facet_wise[row["Facet"]].append(1)
                agreement_store["True"].append(row["Agreement"])
            elif int(row["Selected Answer Nr."]) == 2 and sim_2 > sim_1 and sim_2 > sim_3:
                correctly_assessed.append(1)
                facet_wise[row["Facet"]].append(1)
                agreement_store["True"].append(row["Agreement"])
            elif int(row["Selected Answer Nr."]) == 3 and sim_3 > sim_1 and sim_3 > sim_2:
                correctly_assessed.append(1)
                facet_wise[row["Facet"]].append(1)
                agreement_store["True"].append(row["Agreement"])
            else:
                correctly_assessed.append(0)
                agreement_store["False"].append(row["Agreement"])
                facet_wise[row["Facet"]].append(0)

    print("False:", np.mean(agreement_store["False"]))
    print("True:", np.mean(agreement_store["True"]))
    result_scores = {facet: sum(scores) / len(scores) for facet, scores in facet_wise.items()}
    result_scores["all_facets"] = sum(correctly_assessed) / len(correctly_assessed)
    return result_scores, correctly_assessed, facet_wise


def correlation_for_correctly_labeled(vectors, human_assessment_df: pd.DataFrame, doc_id_mapping: Dict[str, str],
                                      facet_mapping: Dict[str, str], use_sum: bool):
    # reverted_facets = {value: key for key, value in facet_mapping.items()}

    ground_truth = defaultdict(list)
    predicted = defaultdict(list)
    skip_count = 0
    for i, row in human_assessment_df.iterrows():
        book1 = doc_id_mapping[row["Book 1"]]
        book2 = doc_id_mapping[row["Book 2"]]
        book3 = doc_id_mapping[row["Book 3"]]
        if use_sum:
            facet = facet_mapping["total"]
        else:
            facet = facet_mapping[row["Facet"]]
        selection = row["Selection"]

        if selection == "skip" or selection == "unsure":
            skip_count += 1
            continue

        sim_1 = Vectorization.facet_sim(model_vectors=vectors, doc_id_a=book1, doc_id_b=book2, facet_name=facet)
        sim_2 = Vectorization.facet_sim(model_vectors=vectors, doc_id_a=book1, doc_id_b=book3, facet_name=facet)
        sim_3 = Vectorization.facet_sim(model_vectors=vectors, doc_id_a=book2, doc_id_b=book3, facet_name=facet)

        if sim_1 > sim_2 and sim_1 > sim_3:
            pred_label = 1
        elif sim_2 > sim_1 and sim_2 > sim_3:
            pred_label = 2
        elif sim_3 > sim_1 and sim_3 > sim_2:
            pred_label = 3
        else:
            print("warning")
            pred_label = -1

        ground_truth[row["Facet"]].append(int(row["Selected Answer Nr."]))
        ground_truth["all_facets"].append(int(row["Selected Answer Nr."]))
        predicted[row["Facet"]].append(pred_label)
        predicted["all_facets"].append(pred_label)

        # print(row["Facet"], "=", sum(facet_wise[row["Facet"]]))
    print(f"{skip_count} times skipped!")
    result_scores = {}
    for facet, ground_truth_labels in ground_truth.items():
        predicted_labels = predicted[facet]
        corr = stats.spearmanr(ground_truth_labels, predicted_labels)
        spearman = str(f'{abs(corr[0]):.3f}')
        if corr[1] < 0.05:
            spearman = f"*{spearman}"
        result_scores[facet] = spearman

    return result_scores
    # result_scores = {facet: sum(scores) / len(scores) for facet, scores in facet_wise.items()}
    # result_scores["all"] = sum(correctly_assessed) / len(correctly_assessed)
    # return result_scores


def load_vectors_from_properties(number_of_subparts, corpus_size, data_set,
                                 filter_mode, vectorization_algorithm):
    use_sum = False
    if "_sum" in vectorization_algorithm:
        use_sum = True
    vec_path = Vectorization.build_vec_file_name(number_of_subparts,
                                                 corpus_size,
                                                 data_set,
                                                 filter_mode,
                                                 vectorization_algorithm,
                                                 "real",
                                                 allow_combination=True)

    vectors, _ = Vectorization.my_load_doc2vec_format(vec_path)
    return vectors, use_sum


def calculate_vectors(data_set_name: str, vec_algorithms: List[str], filters: List[str]):
    # try:
    #     corpus = Corpus.fast_load(path=os.path.join('corpora', data_set_name), load_entities=False)
    # except FileNotFoundError:
    #     corpus = DataHandler.load_classic_gutenberg_as_corpus()
    #     Preprocesser.annotate_and_save(corpus, corpus_dir=f"corpora/{data_set_name}")
    #     corpus = Corpus.fast_load(path=os.path.join('corpora', data_set_name), load_entities=False)
    for filter_mode in filters:
        corpus = Corpus.fast_load("all",
                                  "no_limit",
                                  data_set_name,
                                  filter_mode,
                                  "real",
                                  load_entities=False)

        for vectorization_algorithm in vec_algorithms:
            use_summation = False
            if "_sum" in vectorization_algorithm:
                vectorization_algorithm = vectorization_algorithm.replace("_sum", "")
                use_summation = True
            vec_file_name = Vectorization.build_vec_file_name('all',
                                                              'no_limit',
                                                              data_set_name,
                                                              filter_mode,
                                                              vectorization_algorithm,
                                                              'real')

            if not os.path.isfile(vec_file_name):
                Vectorizer.algorithm(input_str=vectorization_algorithm,
                                     corpus=corpus,
                                     save_path=vec_file_name,
                                     return_vecs=False)


def evaluate(data_set_name: str, vec_algorithms: List[str], filters: List[str]):
    human_assessment_df = pd.read_csv("results/human_assessment/gutenberg_classic_20/human_assessed_complete.csv")
    print(len(human_assessment_df.index))
    human_assessment_df = human_assessment_df.loc[(human_assessment_df['Selection'] != "unsure")]
    # & (human_assessment_df['Answers'] > 1)
    # human_assessment_df = human_assessment_df.loc[(human_assessment_df['Agreement'] > 0.5)
    #                                               # & (human_assessment_df['Answers'] > 1)
    # ]

    print(len(human_assessment_df.index))
    survey_id2doc_id = {1: "cb_17",
                        2: "cb_2",
                        3: "cb_0",
                        4: "cb_1",
                        5: "cb_3",
                        6: "cb_4",
                        7: "cb_5",
                        8: "cb_6",
                        9: "cb_9",
                        10: "cb_11",
                        11: "cb_12",
                        12: "cb_13",
                        13: "cb_14",
                        14: "cb_15",
                        15: "cb_8",
                        16: "cb_7",
                        17: "cb_10",
                        18: "cb_18",
                        19: "cb_19",
                        20: "cb_16",
                        }
    facets = {"location": "loc", "time": "time", "atmosphere": "atm", "content": "cont", "plot": "plot", "total": ""}

    tuples = []
    correlation_tuples = []
    correctness_table = {}
    correctness_table_facet = {}
    for filter in filters:
        for vec_algorithm in vec_algorithms:
            filtered_dataset = f'{data_set_name}_{filter}'

            # corpus = Corpus.fast_load(path=os.path.join('corpora', f''data_set_name), load_entities=False)
            vecs, use_sum = load_vectors_from_properties(number_of_subparts="all",
                                                         corpus_size="no_limit",
                                                         data_set=data_set_name,
                                                         filter_mode=filter,
                                                         vectorization_algorithm=vec_algorithm)

            corr_scores = correlation_for_correctly_labeled(vectors=vecs, human_assessment_df=human_assessment_df,
                                                            doc_id_mapping=survey_id2doc_id, facet_mapping=facets,
                                                            use_sum=use_sum)
            correlation_tuples.append((filtered_dataset, vec_algorithm, corr_scores["total"],
                                       corr_scores["time"], corr_scores["location"],
                                       corr_scores["plot"], corr_scores["atmosphere"], corr_scores["content"],
                                       corr_scores["all_facets"]))

            scores, cor_ass, facet_wise = get_percentage_of_correctly_labeled(vectors=vecs,
                                                                              human_assessment_df=human_assessment_df,
                                                                              doc_id_mapping=survey_id2doc_id,
                                                                              facet_mapping=facets,
                                                                              use_sum=use_sum)
            correctness_table[vec_algorithm] = cor_ass
            correctness_table_facet[vec_algorithm] = facet_wise

            tuples.append((filtered_dataset, vec_algorithm, scores["total"], scores["time"],
                           scores["location"],
                           scores["plot"],
                           scores["atmosphere"], scores["content"], scores["all_facets"]))
            print((filtered_dataset, vec_algorithm, scores["total"], scores["time"],
                   scores["location"],
                   scores["plot"],
                   scores["atmosphere"], scores["content"], scores["all_facets"]))

    try:
        algo1 = "bert_pt"
        algo2 = "book2vec_adv_dbow_pca"
        true_true = 0
        true_false = 0
        false_true = 0
        false_false = 0
        for e1, e2 in zip(correctness_table[algo1], correctness_table[algo2]):
            if e1 and e2:
                true_true += 1
            elif e1 and not e2:
                true_false += 1
            elif not e1 and e2:
                false_true += 1
            elif not e1 and not e2:
                false_false += 1
            else:
                pass
        table = [[true_true, true_false],
                 [false_true, false_false]]
        print(table)
        print()
        print("Overall")
        mcnemar_sig_text(table)

        # facets = correctness_table_facet[algo1].keys()
        for facet in facets:
            true_true = 0
            true_false = 0
            false_true = 0
            false_false = 0
            for e1, e2 in zip(correctness_table_facet[algo1][facet], correctness_table_facet[algo2][facet]):
                if e1 and e2:
                    true_true += 1
                elif e1 and not e2:
                    true_false += 1
                elif not e1 and e2:
                    false_true += 1
                elif not e1 and not e2:
                    false_false += 1
                else:
                    pass
            table = [[true_true, true_false],
                     [false_true, false_false]]
            print()
            print(table)
            print(facet)
            mcnemar_sig_text(table)

        chi_square_test(correctness_table[algo1], correctness_table[algo1])
    except KeyError:
        pass
    result_df = pd.DataFrame(tuples, columns=["Data set", "Algorithm", "Total", "Time",
                                              "Location", "Plot", "Atmosphere", "Content", "Micro AVG"])
    result_df = result_df.round(3)
    result_df.to_csv("results/human_assessment/performance.csv", index=False)
    print(result_df.to_latex(index=False))

    corr_df = pd.DataFrame(correlation_tuples, columns=["Data set", "Algorithm", "Total", "Time",
                                                        "Location", "Plot", "Atmosphere", "Content", "Micro AVG"])
    corr_df.to_csv("results/human_assessment/correlation_results.csv", index=False)
    print(corr_df.to_latex(index=False))


if __name__ == '__main__':
    data_set = "classic_gutenberg"
    # algorithms = ["avg_wv2doc", "doc2vec", "book2vec", "book2vec_concat"]

    # algorithms = [
    #     # "book2vec_o_time", "book2vec_o_loc", "book2vec_o_atm", "book2vec_o_sty", "book2vec_o_plot", "book2vec_o_raw",
    #     "book2vec",  "book2vec_sum", "book2vec_avg", "book2vec_concat", "book2vec_auto", "book2vec_pca",
    #     "book2vec_dbow",
    #     # "book2vec_dbow_sum", "book2vec_dbow_avg", "book2vec_dbow_concat", "book2vec_dbow_auto",
    #     "book2vec_dbow_pca",
    #     #
    #     "book2vec_wo_raw",
    #     # "book2vec_wo_raw_sum", "book2vec_wo_raw_avg", "book2vec_wo_raw_concat",
    #     # "book2vec_wo_raw_auto",
    #     "book2vec_wo_raw_pca",
    #     # "book2vec_dbow_wo_raw", "book2vec_dbow_wo_raw_sum", "book2vec_dbow_wo_raw_avg",
    #     # "book2vec_dbow_wo_raw_concat", "book2vec_dbow_wo_raw_auto",
    #     "book2vec_dbow_wo_raw_pca",
    #     #
    #     # "book2vec_net_only", "book2vec_net_only_sum", "book2vec_net_only_avg",
    #     # "book2vec_net_only_concat", "book2vec_net_only_auto",
    #     "book2vec_net_only_pca",
    #     # "book2vec_dbow_net_only", "book2vec_dbow_net_only_pca", "book2vec_dbow_net_only_sum",
    #     # "book2vec_dbow_net_only_avg", "book2vec_dbow_net_only_concat",
    #     # "book2vec_dbow_net_only_auto",
    #     "book2vec_dbow_net_only_pca",
    #     #
    #     # "book2vec_net", "book2vec_net_sum", "book2vec_net_avg",
    #     # "book2vec_net_concat", "book2vec_net_auto",
    #     "book2vec_net_pca",
    #     # "book2vec_dbow_net", "book2vec_dbow_net_pca", "book2vec_dbow_net_sum", "book2vec_dbow_net_avg",
    #     # "book2vec_dbow_net_concat", "book2vec_dbow_net_auto",
    #     "book2vec_dbow_net_pca",
    #     #
    #     "book2vec_adv",
    #     # "book2vec_adv_sum", "book2vec_adv_concat", "book2vec_adv_avg", "book2vec_adv_auto",
    #     "book2vec_adv_pca",
    #     # "book2vec_adv_dbow", "book2vec_adv_dbow_sum", "book2vec_adv_dbow_concat", "book2vec_adv_dbow_avg",
    #     # "book2vec_adv_dbow_auto",
    #     "book2vec_adv_dbow_pca",
    #
    #     # "book2vec_adv_dbow_wo_raw_pca",
    #     # "book2vec_adv_dbow_net_wo_raw_pca",
    #
    #     # "book2vec_window_pca",
    #     # "book2vec_dbow_window_pca",
    #     "book2vec_adv_window_pca",
    #     "book2vec_adv_dbow_window_pca",
    #
    #
    #     ]
    # algorithms = ["book2vec_sum", "book2vec"]

    algorithms = [
        # "bow",
        # "avg_wv2doc_restrict10000",
        # "doc2vec",
        # "doc2vec_dbow",
        # "doc2vec_sentence_based_100",
        # "doc2vec_sentence_based_1000",
        # "doc2vec_chunk",
        # "doc2vec_dbow_chunk"
        "bert_pt",
        # "bert_pt_chunk",
        # # "bert_sentence_based_100_pt",
        # "bert_sentence_based_1000_pt",
        # "roberta_pt",
        # "roberta_pt_chunk",
        # "roberta_sentence_based_1000_pt",
        # "xlm_pt",
        # "xlm_pt_chunk",
        # "xlm_sentence_based_1000_pt",
        # "psif",
        # "book2vec_pca",
        # "book2vec_concat",
        # "book2vec_auto",
        # "book2vec_avg",
        #
        # "book2vec_dbow_pca",
        # "book2vec_dbow_concat",
        # "book2vec_dbow_auto",
        "book2vec_dbow_avg",
        "book2vec_dbow_wo_raw_avg",
        "book2vec_dbow_net_only_avg",
        "book2vec_dbow_net_avg",

        # # "book2vec_advn",
        # "book2vec_advn_pca",
        # "book2vec_advn_concat",
        # "book2vec_advn_auto",
        # "book2vec_advn_avg",
        # # "book2vec_advn_dbow",
        "book2vec_advn_dbow_pca",
        # "book2vec_advn_dbow_concat",
        # "book2vec_advn_dbow_auto",
        "book2vec_advn_dbow_avg",
        # "book2vec_bert_pt_pca",
        # "book2vec_bert_pt_window",
        "book2vec_advn_window_pca",
        "book2vec_advn_dbow_window_avg",
    ]

    # algorithms = ["avg_wv2doc", "doc2vec", "doc2vec_dbow",
    #               "doc2vec_sentence_based_100", "doc2vec_sentence_based_1000",
    #               "book2vec", "book2vec_concat", "book2vec_wo_raw", "book2vec_wo_raw_concat",
    #               "book2vec_dbow", "book2vec_dbow_concat",
    #               "book2vec_dbow_wo_raw", "book2vec_dbow_wo_raw_concat",
    #               "book2vec_net", "book2vec_net_concat",
    #               "book2vec_dbow_net", "book2vec_dbow_net_concat",
    #               "book2vec_net_only", "book2vec_net_only_concat",
    #               "book2vec_dbow_net_only", "book2vec_dbow_net_only_concat",
    #               "book2vec_adv", "book2vec_adv_concat", "bow",
    #               "bert", "bert_sentence_based_100", "bert_sentence_based_100_pt", "bert_sentence_based_1000",
    #               "bert_sentence_based_1000_pt",
    #               # "flair_sentence_based_100", "flair_sentence_based_1000",
    #               "roberta_sentence_based_100_pt", "xlm_sentence_based_100_pt",
    #               "psif"
    #               ]
    filters = [
        # "no_filter",
        "specific_words_strict"
    ]

    calculate_vectors(data_set, algorithms, filters)
    evaluate(data_set, algorithms, filters)
