import os
from collections import defaultdict
from typing import Dict, List

import pandas as pd

from corpus_structure import Corpus, DataHandler, Preprocesser
from doc2vec_structures import DocumentKeyedVectors
from vectorization import Vectorizer





def get_percentage_of_correctly_labeled(vectors, human_assessment_df: pd.DataFrame, doc_id_mapping: Dict[str, str],
                                        facet_mapping: Dict[str, str]):
    # reverted_facets = {value: key for key, value in facet_mapping.items()}
    correctly_assessed = []
    facet_wise = defaultdict(list)
    for i, row in human_assessment_df.iterrows():
        book1 = doc_id_mapping[row["Book 1"]]
        book2 = doc_id_mapping[row["Book 2"]]
        book3 = doc_id_mapping[row["Book 3"]]
        facet = facet_mapping[row["Facet"]]
        selection = row["Selection"]

        sim_1 = Vectorizer.facet_sim(model_vectors=vectors, doc_id_a=book1, doc_id_b=book2, facet_name=facet)
        sim_2 = Vectorizer.facet_sim(model_vectors=vectors, doc_id_a=book1, doc_id_b=book3, facet_name=facet)
        sim_3 = Vectorizer.facet_sim(model_vectors=vectors, doc_id_a=book2, doc_id_b=book3, facet_name=facet)

        if selection.split('|')[0] == "1" and sim_1 > sim_2 and sim_1 > sim_3:
            correctly_assessed.append(1)
            facet_wise[row["Facet"]].append(1)
        elif selection.split('|')[0] == "2" and sim_2 > sim_1 and sim_2 > sim_3:
            correctly_assessed.append(1)
            facet_wise[row["Facet"]].append(1)
        elif selection.split('|')[0] == "3" and sim_3 > sim_1 and sim_3 > sim_2:
            correctly_assessed.append(1)
            facet_wise[row["Facet"]].append(1)
        else:
            correctly_assessed.append(0)
            facet_wise[row["Facet"]].append(0)

    result_scores = {facet: sum(scores) / len (scores) for facet, scores in facet_wise.items()}
    result_scores["all"] = sum(correctly_assessed) / len(correctly_assessed)
    return result_scores

def load_vectors_from_properties(number_of_subparts, corpus_size, data_set,
                                 filter_mode, vectorization_algorithm):
    vec_path = Vectorizer.build_vec_file_name(number_of_subparts,
                                              corpus_size,
                                              data_set,
                                              filter_mode,
                                              vectorization_algorithm,
                                              "real")
    # print(vec_path)
    vectors = Vectorizer.my_load_doc2vec_format(vec_path)
    return vectors


def calculate_vectors(data_set_name: str, vec_algorithms: List[str], ):
    try:
        corpus = Corpus.fast_load(path=os.path.join('corpora', data_set_name), load_entities=False)
    except FileNotFoundError:
        corpus = DataHandler.load_classic_gutenberg_as_corpus()
        Preprocesser.annotate_and_save(corpus, corpus_dir=f"corpora/{data_set_name}")
        corpus = Corpus.fast_load(path=os.path.join('corpora', data_set_name), load_entities=False)

    for vectorization_algorithm in vec_algorithms:
        vec_file_name = Vectorizer.build_vec_file_name('',
                                                       '',
                                                       data_set_name,
                                                       'no_filter',
                                                       vectorization_algorithm,
                                                       'real')

        Vectorizer.algorithm(input_str=vectorization_algorithm,
                             corpus=corpus,
                             save_path=vec_file_name,
                             return_vecs=False)


def evaluate(data_set_name: str, vec_algorithms: List[str]):
    human_assessment_df = pd.read_csv("results/human_assessment/human_assessed.csv")
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
    for vec_algorithm in vec_algorithms:
        corpus = Corpus.fast_load(path=os.path.join('corpora', data_set_name), load_entities=False)
        vecs = load_vectors_from_properties(number_of_subparts="",
                                            corpus_size="",
                                            data_set=data_set_name,
                                            filter_mode="no_filter",
                                            vectorization_algorithm=vec_algorithm)

        scores = get_percentage_of_correctly_labeled(vectors=vecs, human_assessment_df=human_assessment_df,
                                                     doc_id_mapping=survey_id2doc_id, facet_mapping=facets)
        print(scores)
        tuples.append((data_set_name, vec_algorithm, scores["all"], scores["total"], scores["time"], scores["location"],
                       scores["plot"],
                       scores["atmosphere"], scores["content"]))

    result_df = pd.DataFrame(tuples, columns=["Data set", "Algorithm", "All", "Total", "Time",
                                              "Location", "Plot", "Atmosphere", "Content"])
    result_df.to_csv("results/human_assessment/performance.csv", index=False)


if __name__ == '__main__':
    data_set = "classic_gutenberg"
    algorithms = ["book2vec_adv", "doc2vec", "avg_wv2doc"]

    # calculate_vectors(data_set, algorithms)
    evaluate(data_set, algorithms)



