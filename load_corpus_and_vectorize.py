import json
import os
import time
from collections import defaultdict
from bs4 import BeautifulSoup
import spacy
from sklearn import metrics
import numpy as np
from tqdm import tqdm

from corpus_structure import DataHandler, Preprocesser, Corpus, Document
import timeit

from vectorization import Vectorizer
from vectorization_utils import Vectorization

if __name__ == "__main__":
    # corpus = DataHandler.load_maharjan_goodreads()
    # Preprocesser.annotate_and_save(corpus, corpus_dir="corpora/goodreads_genres")
    data_set_name = "classic_gutenberg"
    corpus = Corpus.fast_load(path=os.path.join('corpora', data_set_name), load_entities=False)
    # corpus.length_sub_corpora()
    # for doc_id, document in corpus.documents.items():
    #     print(document.genres)
    # corpus = DataHandler.load_classic_gutenberg_as_corpus()
    # Preprocesser.annotate_and_save(corpus, corpus_dir="corpora/classic_gutenberg")
    # data_set_name = "classic_gutenberg"  # "german_series"  #
    # corpus = Corpus.fast_load(path=os.path.join('corpora', data_set_name), load_entities=False)

    # for doc_id, document in corpus.documents.items():
    #     # print(doc_id)
    #     chunks = document.into_chunks(500)
    #     for chunk in chunks:
    #         continue
    #         # print(chunk.length, chunk, chunk.sentences)

    # for doc_id, doc in corpus.documents.items():
    #     doc.load_sentences_from_disk()
    #     print(doc_id, doc.length)
    vectorization_algorithm = "book2vec_adv"
    vec_file_name = Vectorization.build_vec_file_name('all',
                                                      'no_limit',
                                                      data_set_name,
                                                      'no_filter',
                                                      vectorization_algorithm,
                                                      'real')
    # vec_file_name = Vectorization.build_vec_file_name('window',
    #                                                   '',
    #                                                   data_set_name,
    #                                                   'no_filter',
    #                                                   vectorization_algorithm,
    #                                                   'real')
    if not os.path.isfile(vec_file_name):
        Vectorizer.algorithm(input_str=vectorization_algorithm,
                             corpus=corpus,
                             save_path=vec_file_name,
                             return_vecs=False)
