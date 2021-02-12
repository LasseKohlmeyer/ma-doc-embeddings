import os
from typing import List

from tqdm import tqdm

from corpus_structure import Corpus
from vectorization_utils import Vectorization
import pandas as pd


def replace_sim_id(doc_id: str):
    if "_loc" in doc_id:
        doc_id = doc_id.replace("_loc", "")
    elif "_time" in doc_id:
        doc_id = doc_id.replace("_time", "")
    elif "_atm" in doc_id:
        doc_id = doc_id.replace("_atm", "")
    elif "_sty" in doc_id:
        doc_id = doc_id.replace("_sty", "")
    elif "_cont" in doc_id:
        doc_id = doc_id.replace("_cont", "")
    elif "_plot" in doc_id:
        doc_id = doc_id.replace("_plot", "")
    else:
        pass

    return doc_id


def get_neighbors(data_sets: List[str], vector_names: List[str]):
    doc_top_n = 2
    facet_names = [
        "loc",
        "time",
        "atm",
        "sty",
        "cont",
        "plot"
    ]
    is_series_corpus = False
    tuples = []
    for data_set in data_sets:
        corpus = Corpus.fast_load(path=os.path.join('corpora', data_set), load_entities=False)
        for vector_name in tqdm(vector_names, desc="Iterate through embedding types", total=len(vector_names)):
            vec_path = Vectorization.build_vec_file_name("all",
                                                         "no_limit",
                                                         data_set,
                                                         "no_filter",
                                                         vector_name,
                                                         "real",
                                                         allow_combination=True)

            vectors, _ = Vectorization.my_load_doc2vec_format(vec_path)
            for doc_id in corpus.documents.keys():
                for facet_name in facet_names:
                    sim_docs = Vectorization.most_similar_documents(vectors, corpus, positives=doc_id,
                                                                    topn=doc_top_n,
                                                                    feature_to_use=facet_name, print_results=False,
                                                                    series=is_series_corpus)[1:]
                    for i, (sim_doc_id, sim) in enumerate(sim_docs):
                        tuples.append((data_set, vector_name, facet_name, corpus.documents[doc_id], i,
                                       corpus.documents[replace_sim_id(sim_doc_id)], sim))
    df = pd.DataFrame(tuples, columns=["Dataset", "Algorithm", "Facet", "Book", "Rank", "Similar Book", "Similarity"])
    df.to_csv("results/neighbors/neighbors.csv")
    print(df)


if __name__ == '__main__':
    data_set_names = [
        'classic_gutenberg',
        # 'german_series',
    ]
    algorithms = [
        # 'book2vec',
        'book2vec_adv',
        # 'doc2vec',
        # 'avg_wv2doc'
    ]

    get_neighbors(data_set_names, algorithms)
    # data_set_name = "classic_gutenberg"
    # c = Corpus.load_corpus_from_dir_format(os.path.join(f"corpora/{data_set_name}"))
    # # d = TopicModeller.train_lda(c)
    # TopicModeller.get_topic_distribution(c, data_set_name)
