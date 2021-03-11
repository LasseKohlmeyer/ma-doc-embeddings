import os
from collections import defaultdict
import random
from typing import Dict, List

import pandas as pd
from scipy.stats import stats

from corpus_structure import Corpus, DataHandler, Preprocesser
from vectorization import Vectorizer
from vectorization_utils import Vectorization


def get_percentage_of_correctly_labeled(vectors, human_assessment_df: pd.DataFrame, doc_id_mapping: Dict[str, str],
                                        facet_mapping: Dict[str, str]):
    # reverted_facets = {value: key for key, value in facet_mapping.items()}
    correctly_assessed = []
    facet_wise = defaultdict(list)
    random_baseline = False
    skip_count = 0
    for i, row in human_assessment_df.iterrows():
        book1 = doc_id_mapping[row["Book 1"]]
        book2 = doc_id_mapping[row["Book 2"]]
        book3 = doc_id_mapping[row["Book 3"]]
        facet = facet_mapping[row["Facet"]]
        selection = row["Selection"]
        if selection == "skip" or selection == "unsure":
            # print("skipped")
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

            # print(selection, selection.split('|')[0], book1, book2, book3)
            # print()
            if int(row["Selected Answer Nr."]) == 1 and sim_1 > sim_2 and sim_1 > sim_3:
                correctly_assessed.append(1)
                facet_wise[row["Facet"]].append(1)
            elif int(row["Selected Answer Nr."]) == 2 and sim_2 > sim_1 and sim_2 > sim_3:
                correctly_assessed.append(1)
                facet_wise[row["Facet"]].append(1)
            elif int(row["Selected Answer Nr."]) == 3 and sim_3 > sim_1 and sim_3 > sim_2:
                correctly_assessed.append(1)
                facet_wise[row["Facet"]].append(1)
            else:
                correctly_assessed.append(0)
                facet_wise[row["Facet"]].append(0)
            # print(row["Facet"], "=", sum(facet_wise[row["Facet"]]))

    # for facet, scores in facet_wise.items():
    #     print(facet, sum(scores), scores)
    # print(f"{skip_count} times skipped!")
    result_scores = {facet: sum(scores) / len(scores) for facet, scores in facet_wise.items()}
    result_scores["all_facets"] = sum(correctly_assessed) / len(correctly_assessed)
    return result_scores


def correlation_for_correctly_labeled(vectors, human_assessment_df: pd.DataFrame, doc_id_mapping: Dict[str, str],
                                        facet_mapping: Dict[str, str]):
    # reverted_facets = {value: key for key, value in facet_mapping.items()}

    ground_truth = defaultdict(list)
    predicted = defaultdict(list)
    skip_count = 0
    for i, row in human_assessment_df.iterrows():
        book1 = doc_id_mapping[row["Book 1"]]
        book2 = doc_id_mapping[row["Book 2"]]
        book3 = doc_id_mapping[row["Book 3"]]
        facet = facet_mapping[row["Facet"]]
        selection = row["Selection"]

        if selection == "skip" or selection == "unsure":
            # print("skipped")
            skip_count += 1
            continue

        sim_1 = Vectorization.facet_sim(model_vectors=vectors, doc_id_a=book1, doc_id_b=book2, facet_name=facet)
        sim_2 = Vectorization.facet_sim(model_vectors=vectors, doc_id_a=book1, doc_id_b=book3, facet_name=facet)
        sim_3 = Vectorization.facet_sim(model_vectors=vectors, doc_id_a=book2, doc_id_b=book3, facet_name=facet)

        # print(selection, selection.split('|')[0], book1, book2, book3)
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
        # print(facet)
        # print(ground_truth_labels)
        # print(predicted_labels)
        corr = stats.spearmanr(ground_truth_labels, predicted_labels)
        spearman = str(abs(corr[0]))
        if corr[1] < 0.05:
            spearman = f"*{spearman}"
        result_scores[facet] = spearman
        # print(facet, corr)
    return result_scores
    # result_scores = {facet: sum(scores) / len(scores) for facet, scores in facet_wise.items()}
    # result_scores["all"] = sum(correctly_assessed) / len(correctly_assessed)
    # return result_scores


def load_vectors_from_properties(number_of_subparts, corpus_size, data_set,
                                 filter_mode, vectorization_algorithm):
    vec_path = Vectorization.build_vec_file_name(number_of_subparts,
                                                 corpus_size,
                                                 data_set,
                                                 filter_mode,
                                                 vectorization_algorithm,
                                                 "real",
                                                 allow_combination=True)
    # print(vec_path)
    vectors, _ = Vectorization.my_load_doc2vec_format(vec_path)
    return vectors


def calculate_vectors(data_set_name: str, vec_algorithms: List[str], filters: List[str]):
    try:
        corpus = Corpus.fast_load(path=os.path.join('corpora', data_set_name), load_entities=False)
    except FileNotFoundError:
        corpus = DataHandler.load_classic_gutenberg_as_corpus()
        Preprocesser.annotate_and_save(corpus, corpus_dir=f"corpora/{data_set_name}")
        corpus = Corpus.fast_load(path=os.path.join('corpora', data_set_name), load_entities=False)

    for filter in filters:
        for vectorization_algorithm in vec_algorithms:
            vec_file_name = Vectorization.build_vec_file_name('all',
                                                              'no_limit',
                                                              data_set_name,
                                                              filter,
                                                              vectorization_algorithm,
                                                              'real')

            if not os.path.isfile(vec_file_name):
                print(vec_file_name)
                Vectorizer.algorithm(input_str=vectorization_algorithm,
                                     corpus=corpus,
                                     save_path=vec_file_name,
                                     return_vecs=False)


def evaluate(data_set_name: str, vec_algorithms: List[str], filters: List[str]):
    human_assessment_df = pd.read_csv("results/human_assessment/gutenberg_classic_20/human_assessed.csv")
    print(len(human_assessment_df.index))
    human_assessment_df = human_assessment_df.loc[(human_assessment_df['Confidence'] > 0.5)
                                                  # & (human_assessment_df['Answers'] > 1)
    ]

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
    facets = {"location": "loc", "time": " time", "atmosphere": "atm", "content": "cont", "plot": "plot", "total": ""}

    tuples = []
    correlation_tuples = []
    for filter in filters:
        for vec_algorithm in vec_algorithms:
            filtered_dataset = f'{data_set_name}_{filter}'
            # corpus = Corpus.fast_load(path=os.path.join('corpora', f''data_set_name), load_entities=False)
            vecs = load_vectors_from_properties(number_of_subparts="all",
                                                corpus_size="no_limit",
                                                data_set=data_set_name,
                                                filter_mode=filter,
                                                vectorization_algorithm=vec_algorithm)

            corr_scores = correlation_for_correctly_labeled(vectors=vecs, human_assessment_df=human_assessment_df,
                                                            doc_id_mapping=survey_id2doc_id, facet_mapping=facets)
            correlation_tuples.append((data_set_name, vec_algorithm, corr_scores["all_facets"], corr_scores["total"],
                                       corr_scores["time"], corr_scores["location"],
                                       corr_scores["plot"], corr_scores["atmosphere"], corr_scores["content"]))

            scores = get_percentage_of_correctly_labeled(vectors=vecs, human_assessment_df=human_assessment_df,
                                                         doc_id_mapping=survey_id2doc_id, facet_mapping=facets)
            # print(scores)
            tuples.append((filtered_dataset, vec_algorithm, scores["all_facets"], scores["total"], scores["time"],
                           scores["location"],
                           scores["plot"],
                           scores["atmosphere"], scores["content"]))
            print((filtered_dataset, vec_algorithm, scores["all_facets"], scores["total"], scores["time"],
                           scores["location"],
                           scores["plot"],
                           scores["atmosphere"], scores["content"]))

    result_df = pd.DataFrame(tuples, columns=["Data set", "Algorithm", "All Facets", "Total", "Time",
                                              "Location", "Plot", "Atmosphere", "Content"])
    result_df.to_csv("results/human_assessment/performance.csv", index=False)
    print(result_df)

    corr_df = pd.DataFrame(correlation_tuples, columns=["Data set", "Algorithm", "All Facets", "Total", "Time",
                                                        "Location", "Plot", "Atmosphere", "Content"])
    corr_df.to_csv("results/human_assessment/correlation_results.csv", index=False)
    print(corr_df)


if __name__ == '__main__':
    data_set = "classic_gutenberg"
    algorithms = ["avg_wv2doc", "doc2vec", "doc2vec_dbow",
                  "doc2vec_sentence_based_100", "doc2vec_sentence_based_1000",
                  "book2vec", "book2vec_concat",
                  "book2vec_dbow", "book2vec_dbow_concat",
                  "book2vec_net", "book2vec_net_concat",
                  "book2vec_dbow_net", "book2vec_dbow_net_concat",
                  "book2vec_net_only", "book2vec_net_only_concat",
                  "book2vec_dbow_net_only", "book2vec_dbow_net_only_concat",
                  "book2vec_adv", "book2vec_adv_concat", "bow",
                  "bert", "bert_sentence_based_100", "bert_sentence_based_100_pt", "bert_sentence_based_1000",
                  "bert_sentence_based_1000_pt",
                  # "flair_sentence_based_100", "flair_sentence_based_1000",
                  "roberta_sentence_based_100_pt", "xlm_sentence_based_100_pt",
                  ]
    filters = ["no_filter",
               "specific_words_strict"
               ]

    calculate_vectors(data_set, algorithms, filters)
    evaluate(data_set, algorithms, filters)

    # good seeds: 10
    # bad:
