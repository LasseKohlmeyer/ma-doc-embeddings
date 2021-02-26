import logging
import os
import time
from collections import defaultdict
from typing import Union, List, Dict

import nltk
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import pandas as pd
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile, datapath
from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm
import torch
from transformers import TFAutoModel, AutoTokenizer, AdamW, AutoModel

from corpus_iterators import CorpusSentenceIterator, \
    CorpusDocumentIterator, CorpusTaggedDocumentIterator, CorpusTaggedFacetIterator, \
    write_doc_based_aspect_frequency_analyzis, FlairDocumentIterator, FlairFacetIterator, CorpusTaggedSentenceIterator, \
    CorpusPlainDocumentIterator, FlairSentenceDocumentIterator
from flair_connector import FlairConnector
from text_summarisation import Summarizer
from topic_modelling import TopicModeller
from corpus_structure import Corpus, ConfigLoader, Language
from vectorization_utils import Vectorization

config = ConfigLoader.get_config()


def robust_vec_loading(pretrained_emb_path: str = None, binary: bool = False):
    logging.info(f'Load pretrained embeddings from {pretrained_emb_path}')
    if pretrained_emb_path is None:
        return None
    try:
        model = KeyedVectors.load_word2vec_format(pretrained_emb_path, binary=binary)
    except ValueError:
        glove_file = datapath(pretrained_emb_path)
        tmp_file = get_tmpfile("test_word2vec.txt")
        _ = glove2word2vec(glove_file, tmp_file)
        model = KeyedVectors.load_word2vec_format(tmp_file)
    logging.info('load completed')
    return model


class Vectorizer:
    workers = 1
    seed = 42
    window = 10
    min_count = 0
    epochs = 20
    dim = 300
    pretrained_emb_path = config["embeddings"]["pretrained"]
    # "E:/embeddings/glove.6B.300d.txt" # "E:/embeddings/google300.txt"
    pretrained_emb = None
    pretrained_emb_path_german = config["embeddings"]["pretrained_german"]
    # "E:/embeddings/glove.6B.300d.txt" # "E:/embeddings/google300.txt"
    pretrained_emb_german = None

    @staticmethod
    def algorithm(input_str: str, corpus: Corpus, save_path: str = "models/",
                  return_vecs: bool = False, chunk_len: int = None):
        if "_o_" in input_str:
            return

        concat = False
        if "_concat" in input_str:
            concat = True
            input_str = input_str.replace("_concat", "")

        dim = None
        # if "dim" in input_str and input_str[-1].isdigit():
        #     dim = int(input_str.split('_')[-1])
        #     input_str = input_str.replace(f"_dim{dim}", "")

        facets_of_chunks = False
        if chunk_len is None and "_chunk" in input_str:
            if "doc2vec" in input_str:
                chunk_len = 10000
            elif "longformer" in input_str:
                chunk_len = 4096
            else:
                chunk_len = 512

            if "_chunk_facet" in input_str:
                facets_of_chunks = True
                input_str = input_str.replace("_facet", "")

        window = 0
        if "_window" in input_str:
            window = 5
            input_str = input_str.replace("_window", "")

        input_str = input_str.replace("_chunk", "")
        if input_str == "avg_wv2doc":
            return Vectorizer.avg_wv2doc(corpus, save_path, return_vecs=return_vecs, dimension=dim)
        elif input_str == "avg_wv2doc_restrict10000":
            return Vectorizer.avg_wv2doc(corpus, save_path, return_vecs=return_vecs,
                                         restrict_to=10000, dimension=dim)
        elif input_str == "avg_wv2doc_untrained":
            return Vectorizer.avg_wv2doc(corpus, save_path, return_vecs=return_vecs, without_training=True,
                                         dimension=dim)
        elif input_str == "avg_wv2doc_pretrained":
            return Vectorizer.avg_wv2doc(corpus, save_path, return_vecs=return_vecs,
                                         dimension=dim, pretrained=True)
        elif input_str == "bow":
            return Vectorizer.bow(corpus, save_path, return_vecs=return_vecs)
        elif input_str == "doc2vec":
            return Vectorizer.doc2vec(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len, dimension=dim)
        elif input_str == "doc2vec_sentence_based":
            return Vectorizer.doc2vec(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len, dimension=dim,
                                      sentence_based=True)
        elif input_str == "doc2vec_sentence_based_100":
            return Vectorizer.doc2vec(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len, dimension=dim,
                                      sentence_based=True, sentence_nr=100)
        elif input_str == "doc2vec_sentence_based_1000":
            return Vectorizer.doc2vec(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len, dimension=dim,
                                      sentence_based=True, sentence_nr=1000)
        elif input_str == "doc2vec_dim50":
            return Vectorizer.doc2vec(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len, dimension=50)
        elif input_str == "doc2vec_dim100":
            return Vectorizer.doc2vec(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len, dimension=100)
        elif input_str == "doc2vec_dim300":
            return Vectorizer.doc2vec(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len, dimension=300)
        elif input_str == "doc2vec_dim500":
            return Vectorizer.doc2vec(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len, dimension=500)
        elif input_str == "doc2vec_dim700":
            return Vectorizer.doc2vec(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len, dimension=700)
        elif input_str == "doc2vec_dim900":
            return Vectorizer.doc2vec(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len, dimension=900)
        elif input_str == "doc2vec_untrained":
            return Vectorizer.doc2vec(corpus, save_path, return_vecs=return_vecs, without_training=True,
                                      chunk_len=chunk_len, dimension=dim)
        elif input_str == "doc2vec_pretrained":
            return Vectorizer.doc2vec(corpus, save_path, return_vecs=return_vecs,
                                      chunk_len=chunk_len, dimension=dim, pretrained=True)
        # elif input_str == "longformer" or "longformer_untuned" or "untuned_longformer":
        #     return Vectorizer.longformer_untuned(corpus, save_path, return_vecs=return_vecs)
        # elif input_str == "longformer_tuned" or "tuned_longformer":
        #     return Vectorizer.longformer_tuned(corpus, save_path, return_vecs=return_vecs)
        elif input_str == "book2vec_simple" or input_str == "book2vec":
            return Vectorizer.book2vec_simple(corpus, save_path, return_vecs=return_vecs,
                                              disable_aspects=['plot', 'cont'], chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_simple_net" or input_str == "book2vec_net":
            return Vectorizer.book2vec_simple(corpus, save_path, return_vecs=return_vecs,
                                              disable_aspects=['plot', 'cont'], chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim,
                                              use_dictionary_lookup=True)
        elif input_str == "book2vec_simple_pretained" or input_str == "book2vec_pretrained":
            return Vectorizer.book2vec_simple(corpus, save_path, return_vecs=return_vecs,
                                              disable_aspects=['plot', 'cont'], chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim,
                                              pretrained=True)
        elif input_str == "book2vec_simple_dim50" or input_str == "book2vec_dim50":
            return Vectorizer.book2vec_simple(corpus, save_path, return_vecs=return_vecs,
                                              disable_aspects=['plot', 'cont'], chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=50)
        elif input_str == "book2vec_simple_dim100" or input_str == "book2vec_dim100":
            return Vectorizer.book2vec_simple(corpus, save_path, return_vecs=return_vecs,
                                              disable_aspects=['plot', 'cont'], chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=100)
        elif input_str == "book2vec_simple_dim300" or input_str == "book2vec_dim300":
            return Vectorizer.book2vec_simple(corpus, save_path, return_vecs=return_vecs,
                                              disable_aspects=['plot', 'cont'], chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=300)
        elif input_str == "book2vec_simple_dim500" or input_str == "book2vec_dim500":
            return Vectorizer.book2vec_simple(corpus, save_path, return_vecs=return_vecs,
                                              disable_aspects=['plot', 'cont'], chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=500)
        elif input_str == "book2vec_simple_dim700" or input_str == "book2vec_dim700":
            return Vectorizer.book2vec_simple(corpus, save_path, return_vecs=return_vecs,
                                              disable_aspects=['plot', 'cont'], chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=700)
        elif input_str == "book2vec_simple_dim900" or input_str == "book2vec_dim900":
            return Vectorizer.book2vec_simple(corpus, save_path, return_vecs=return_vecs,
                                              disable_aspects=['plot', 'cont'], chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=900)

        elif input_str == "book2vec_wo_raw":
            return Vectorizer.book2vec_simple(corpus, save_path,
                                              disable_aspects=['raw'], return_vecs=return_vecs, chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_wo_loc":
            return Vectorizer.book2vec_simple(corpus, save_path,
                                              disable_aspects=['loc'], return_vecs=return_vecs, chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_wo_time":
            return Vectorizer.book2vec_simple(corpus, save_path,
                                              disable_aspects=['time'], return_vecs=return_vecs, chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_wo_sty":
            return Vectorizer.book2vec_simple(corpus, save_path,
                                              disable_aspects=['sty'], return_vecs=return_vecs, chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_wo_atm":
            return Vectorizer.book2vec_simple(corpus, save_path,
                                              disable_aspects=['atm'], return_vecs=return_vecs, chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_simple_untrained" or input_str == "book2vec_untrained":
            return Vectorizer.book2vec_simple(corpus, save_path, return_vecs=return_vecs,
                                              without_training=True, chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv":
            return Vectorizer.book2vec_adv(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv_net":
            return Vectorizer.book2vec_adv(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim,
                                           use_dictionary_lookup=True)
        elif input_str == "book2vec_adv_pretrained":
            return Vectorizer.book2vec_adv(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim,
                                           pretrained=True)
        elif input_str == "book2vec_adv_wo_raw":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['raw'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv_wo_loc":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['loc'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv_wo_time":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['time'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv_wo_sty":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['sty'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv_wo_atm":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['atm'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv_wo_plot":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['plot'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv_wo_cont":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['cont'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv_wo_raw":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['raw'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv_wo_loc":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['loc'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv_wo_time":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['time'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv_wo_sty":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['sty'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv_wo_atm":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['atm'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv_wo_plot":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['plot'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_adv_wo_cont":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['cont'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        # elif input_str == "book2vec_adv_w_raw":
        #     return Vectorizer.book2vec_adv(corpus, save_path,
        #                                    disable_aspects=['raw'], return_vecs=return_vecs, enable_mode=True)
        # elif input_str == "book2vec_adv_w_loc":
        #     return Vectorizer.book2vec_adv(corpus, save_path,
        #                                    disable_aspects=['loc'], return_vecs=return_vecs, enable_mode=True)
        # elif input_str == "book2vec_adv_w_time":
        #     return Vectorizer.book2vec_adv(corpus, save_path,
        #                                    disable_aspects=['time'], return_vecs=return_vecs, enable_mode=True)
        # elif input_str == "book2vec_adv_w_sty":
        #     return Vectorizer.book2vec_adv(corpus, save_path,
        #                                    disable_aspects=['sty'], return_vecs=return_vecs, enable_mode=True)
        # elif input_str == "book2vec_adv_w_atm":
        #     return Vectorizer.book2vec_adv(corpus, save_path,
        #                                    disable_aspects=['atm'], return_vecs=return_vecs, enable_mode=True)
        # elif input_str == "book2vec_adv_w_plot":
        #     return Vectorizer.book2vec_adv(corpus, save_path,
        #                                    disable_aspects=['plot'], return_vecs=return_vecs, enable_mode=True)
        # elif input_str == "book2vec_adv_w_cont":
        #     return Vectorizer.book2vec_adv(corpus, save_path,
        #                                    disable_aspects=['cont'], return_vecs=return_vecs, enable_mode=True)
        elif input_str == "book2vec_w2v":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['cont'],
                                           return_vecs=return_vecs,
                                           algorithm="avg_w2v",
                                           chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "book2vec_bert":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['cont'],
                                           return_vecs=return_vecs,
                                           algorithm="transformer",
                                           chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window, dimension=dim)
        elif input_str == "random_aspect2vec" or input_str == "random":
            return Vectorizer.random_aspect2vec(corpus, save_path, return_vecs=return_vecs,
                                                algorithm="doc2vec", dimension=dim)
        elif input_str == "glove":
            return Vectorizer.flair(corpus, "glove", "pool", save_path, return_vecs=return_vecs, chunk_len=chunk_len)
        elif input_str == "glove_rnn":
            return Vectorizer.flair(corpus, "glove", "rnn", save_path, return_vecs=return_vecs, chunk_len=chunk_len)
        elif input_str == "fasttext":
            return Vectorizer.flair(corpus, "fasttext", "pool", save_path, return_vecs=return_vecs, chunk_len=chunk_len)
        elif input_str == "fasttext_rnn":
            return Vectorizer.flair(corpus, "fasttext", "rnn", save_path, return_vecs=return_vecs, chunk_len=chunk_len)
        elif input_str == "bert":
            return Vectorizer.flair(corpus, None, "bert", save_path, return_vecs=return_vecs, chunk_len=chunk_len)
        elif input_str == "bert_sentence_based":
            return Vectorizer.flair(corpus, None, "bert", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True)
        elif input_str == "bert_sentence_based_100":
            return Vectorizer.flair(corpus, None, "bert", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True, sentence_nr=100)
        elif input_str == "bert_sentence_based_1000":
            return Vectorizer.flair(corpus, None, "bert", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True, sentence_nr=1000)
        elif input_str == "longformer":
            return Vectorizer.flair(corpus, None, "longformer", save_path, return_vecs=return_vecs, chunk_len=chunk_len)
        elif input_str == "flair":
            return Vectorizer.flair(corpus, None, "flair", save_path, return_vecs=return_vecs, chunk_len=chunk_len)
        elif input_str == "flair_sentence_based":
            return Vectorizer.flair(corpus, None, "flair", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True)
        elif input_str == "flair_sentence_based_100":
            return Vectorizer.flair(corpus, None, "flair", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True, sentence_nr=100)
        elif input_str == "flair_sentence_based_1000":
            return Vectorizer.flair(corpus, None, "flair", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True, sentence_nr=1000)
        elif input_str == "stacked_flair":
            return Vectorizer.flair(corpus, None, "stacked_flair", save_path, return_vecs=return_vecs,
                                    chunk_len=chunk_len)
        elif input_str == "topic_vec" or input_str == "topic_vecs" or input_str == "topic2vec":
            Vectorizer.topiv_vecs(corpus, save_path=save_path, return_vecs=return_vecs)
        elif input_str == "bert_sentence_based_pt":
            return Vectorizer.flair(corpus, None, "bert", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True, pretuned=True)
        elif input_str == "bert_sentence_based_100_pt":
            return Vectorizer.flair(corpus, None, "bert", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True, sentence_nr=100, pretuned=True)
        elif input_str == "bert_sentence_based_1000_pt":
            return Vectorizer.flair(corpus, None, "bert", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True, sentence_nr=1000, pretuned=True)
        elif input_str == "roberta_sentence_based_pt":
            return Vectorizer.flair(corpus, None, "roberta", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True, pretuned=True)
        elif input_str == "roberta_sentence_based_100_pt":
            return Vectorizer.flair(corpus, None, "roberta", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True, sentence_nr=100, pretuned=True)
        elif input_str == "roberta_sentence_based_1000_pt":
            return Vectorizer.flair(corpus, None, "roberta", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True, sentence_nr=1000, pretuned=True)
        elif input_str == "xlm_sentence_based_pt":
            return Vectorizer.flair(corpus, None, "xlm", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True, pretuned=True)
        elif input_str == "xlm_sentence_based_100_pt":
            return Vectorizer.flair(corpus, None, "xlm", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True, sentence_nr=100, pretuned=True)
        elif input_str == "xlm_sentence_based_1000_pt":
            return Vectorizer.flair(corpus, None, "xlm", save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                    sentence_based=True, sentence_nr=1000, pretuned=True)
        else:
            raise UserWarning(f"fUnknown input string {input_str}!")

    @classmethod
    def word2vec_base(cls,
                      preprocessed_sentences: Union[List[List[str]], CorpusSentenceIterator],
                      preprocessed_documents: Union[List[str], CorpusDocumentIterator, CorpusTaggedFacetIterator],
                      doc_ids,
                      without_training: bool,
                      restrict_to: int = None,
                      dimension: int = None,
                      language: Language = Language.EN,
                      pretrained: bool = False):
        if dimension is None:
            dimension = cls.dim
        print(f'use pretrained = {pretrained} for {language}')
        if pretrained:
            if language == Language.EN:
                if cls.pretrained_emb is None:
                    cls.pretrained_emb = robust_vec_loading(cls.pretrained_emb_path, binary=False)

                model = cls.pretrained_emb
            elif language == Language.DE:
                if cls.pretrained_emb_german is None:
                    cls.pretrained_emb_german = robust_vec_loading(cls.pretrained_emb_path_german, binary=True)

                model = cls.pretrained_emb_german
            else:
                raise UserWarning("Not supported language for pretraining")
        else:
            # model = Word2Vec(preprocessed_sentences, size=cls.dim, window=cls.window, min_count=cls.min_count,
            #                  workers=cls.workers, iter=cls.epochs, seed=cls.seed)
            model = Word2Vec(size=dimension, window=cls.window, min_count=cls.min_count,
                             workers=cls.workers, seed=cls.seed)
            model.build_vocab(preprocessed_sentences)
            if not without_training:
                model.train(preprocessed_sentences, total_examples=model.corpus_count, epochs=cls.epochs)

        docs_dict = {}
        for doc_id, doc in zip(doc_ids, preprocessed_documents):
            vector = []
            if isinstance(doc, TaggedDocument):
                doc = doc.words
                # print(doc_id, doc)
            if len(doc) == 0:
                continue
            if restrict_to:
                doc = doc[:restrict_to]
            for token in doc:
                # print(token, model.wv.vocab[token])
                try:
                    vec = model.wv[token]
                    vector.append(vec)
                except KeyError:
                    try:
                        vec = model.wv[token.lower()]
                        vector.append(vec)
                    except KeyError:
                        logging.error(f'KeyError Error for {doc_id} and {token}')
            # print(doc_id, doc, vector)
            try:
                vector = sum(np.array(vector)) / len(vector)
                docs_dict[doc_id] = vector
            except ZeroDivisionError:
                logging.error(f'ZeroDivision Error for {doc_id}')
                raise UserWarning(f"ZeroDevision Error for {doc_id}")

        words_dict = {word: model.wv[word] for word in model.wv.vocab}
        return model, words_dict, docs_dict

    @classmethod
    def avg_sim_prefix_doc_ids(cls, input_doc_vectors):
        docs_dict = {}
        completed_ids = set()
        for doc_id, doc in input_doc_vectors.items():
            prefix_doc_id = '_'.join(doc_id.split('_')[:-1])
            doc_vectors = [doc]
            completed_ids.add(doc_id)

            for doc_id_2, doc_2 in input_doc_vectors.items():
                if str(doc_id_2).startswith(f'{prefix_doc_id}_') and doc_id_2 not in completed_ids:
                    completed_ids.add(doc_id_2)
                    doc_vectors.append(doc_2)
                    # print(prefix_doc_id, doc_2, len(doc_vectors))
            try:
                vector = sum(np.array(doc_vectors)) / len(doc_vectors)
                docs_dict[prefix_doc_id] = vector
            except ZeroDivisionError:
                logging.error(f'ZeroDivision Error for {doc_id}')
                raise UserWarning(f"ZeroDevision Error for {doc_id}")

        return docs_dict

    @classmethod
    def doc2vec_base(cls, documents: Union[List[str], CorpusTaggedDocumentIterator, CorpusTaggedFacetIterator],
                     without_training: bool, chunk_len: int = None, dimension: int = None, sentence_based: bool = False,
                     language: Language = Language.EN,
                     pretrained: bool = False):
        # model = Doc2Vec(documents, vector_size=100, window=10, min_count=2, workers=4, epochs=20)
        # model = Doc2Vec(documents, vector_size=cls.dim, window=cls.window, min_count=cls.min_count,
        #                 workers=cls.workers, epochs=cls.epochs, pretrained_emb=cls.pretrained_emb_path, seed=cls.seed)
        # print('train')
        if dimension is None:
            dimension = cls.dim

        embedding_path = None
        print(f'use pretrained = {pretrained} for {language}')
        if pretrained:
            embedding_path = cls.pretrained_emb_path
            if language == Language.DE:
                embedding_path = cls.pretrained_emb_path_german

        model = Doc2Vec(vector_size=dimension, min_count=cls.min_count, epochs=cls.epochs,
                        pretrained_emb=embedding_path, seed=cls.seed, workers=cls.workers,
                        window=cls.window)

        model.build_vocab(documents)
        if not without_training:
            model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        # print(model.docvecs.doctags)
        # for tag in model.docvecs.doctags:
        #     if not (tag.endswith('_time') or tag.endswith('_loc')):
        #         new_vec = model.docvecs[tag] + model.docvecs[f'{tag}_time'] + model.docvecs[f'{tag}_loc']
        #         print(tag)
        #         print(model.docvecs[tag])
        #         print(new_vec)
        #     # print(model.docvecs[tag])
        # aspect_string = ''.join(disable_aspects)
        # print(model.docvecs.doctags)
        words_dict, docs_dict = Vectorizer.model2dict(model)

        if chunk_len:
            docs_dict.update(cls.avg_sim_prefix_doc_ids(docs_dict))
        if sentence_based:
            docs_dict = cls.avg_sim_prefix_doc_ids(docs_dict)

        return model, words_dict, docs_dict

    @classmethod
    def flair_base(cls, documents: Union[List[str], FlairFacetIterator, FlairDocumentIterator],
                   word_embedding_base: str = None, document_embedding: str = None, chunk_len: int = None,
                   sentence_based: bool = None, pretuned: bool = False):
        """

        :param pretuned:
        :param sentence_based:
        :param chunk_len: length of chunks
        :param documents: input documents
        :param word_embedding_base: - glove: 'glove', (only en), - fasttext: 'en', 'de'
        :param document_embedding:  pool vs rnn for w2v mode - bert: 'bert', 'bert-de'  - 'longformer' (only en) -
        'flair', 'stacked-flair', 'flair-de', 'stacked-flair-de'
        """
        flair_instance = FlairConnector(word_embedding_base=word_embedding_base, document_embedding=document_embedding,
                                        pretuned=pretuned)
        print(sentence_based, len(documents))

        start = time.time()
        docs_dict = flair_instance.embedd_documents(documents)
        end = time.time()
        print("time:", end-start)

        if chunk_len:
            docs_dict.update(cls.avg_sim_prefix_doc_ids(docs_dict))
        if sentence_based:
            docs_dict = cls.avg_sim_prefix_doc_ids(docs_dict)

        return docs_dict

    @classmethod
    def avg_wv2doc(cls, corpus: Corpus, save_path: str = "models/", return_vecs: bool = True,
                   without_training: bool = False, restrict_to: int = None, dimension: int = None,
                   pretrained: bool = False):
        # Preprocesser.preprocess(return_in_sentence_format=True)
        # print('sents', preprocessed_sentences)
        # print(preprocessed_documents)
        _, doc_ids = corpus.get_texts_and_doc_ids()
        # preprocessed_sentences = corpus.get_flat_corpus_sentences()
        # preprocessed_documents = corpus.get_flat_document_tokens()
        # preprocessed_sentences = corpus.get_flat_corpus_sentences(generator=True)
        # preprocessed_documents = corpus.get_flat_document_tokens(generator=True)

        preprocessed_sentences = CorpusSentenceIterator(corpus)
        preprocessed_documents = CorpusDocumentIterator(corpus)

        # for d in preprocessed_documents:
        #     print(d[:10])
        # print(preprocessed_documents)
        model, words_dict, docs_dict = cls.word2vec_base(preprocessed_sentences,
                                                         preprocessed_documents,
                                                         doc_ids,
                                                         without_training,
                                                         restrict_to,
                                                         dimension,
                                                         language=corpus.language,
                                                         pretrained=pretrained)

        return Vectorization.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=words_dict,
                                                   return_vecs=return_vecs)

    @classmethod
    def doc2vec(cls, corpus: Corpus, save_path: str = "models/", return_vecs: bool = True,
                without_training: bool = False, chunk_len: int = None, dimension: int = None,
                sentence_based: bool = False,
                sentence_nr: int = None,
                pretrained: bool = False):
        # documents = [TaggedDocument(doc, [i])
        #              for i, doc in enumerate(Preprocesser.tokenize(corpus.get_texts_and_doc_ids()))]
        # documents = [TaggedDocument(Preprocesser.tokenize(document.text), [doc_id])
        #              for doc_id, document in corpus.documents.items()]

        if not sentence_based:
            documents = CorpusTaggedDocumentIterator(corpus, chunk_len=chunk_len)
        else:
            documents = CorpusTaggedSentenceIterator(corpus, sentence_nr=sentence_nr)

        model, words_dict, docs_dict = cls.doc2vec_base(documents, without_training, chunk_len=chunk_len,
                                                        dimension=dimension, sentence_based=sentence_based,
                                                        language=corpus.language,
                                                        pretrained=pretrained)

        return Vectorization.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=words_dict,
                                                   return_vecs=return_vecs)

    @classmethod
    def bow(cls, corpus: Corpus, save_path: str = "models/", return_vecs: bool = True):
        documents = CorpusPlainDocumentIterator(corpus)
        # max_df = (len(documents) / 2)
        if corpus.language == Language.DE:
            stopwords = nltk.corpus.stopwords.words('german')
        else:
            stopwords = nltk.corpus.stopwords.words('english')
        bow_model = CountVectorizer(ngram_range=(1, 1),  # to use bigrams ngram_range=(2,2)
                                    stop_words=None, max_features=30000, min_df=2)
        bow_data = bow_model.fit_transform(documents)
        cv_dataframe = pd.DataFrame(bow_data.toarray(), columns=bow_model.get_feature_names())
        df = cv_dataframe.transpose()
        df.columns = documents.doc_ids
        docs_dict = {doc_id: np.array(df[doc_id]) for doc_id in df.columns}

        return Vectorization.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=None,
                                                   return_vecs=return_vecs)


    @classmethod
    def longformer_untuned(cls, corpus: Corpus, save_path: str = "models/", return_vecs: bool = True):
        _, doc_ids = corpus.get_texts_and_doc_ids()

        model_name = "allenai/longformer-base-4096"  # "bert-base-uncased"
        model = TFAutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        documents = corpus.get_flat_documents()

        tokenized_docs = tokenizer(
            documents,
            padding=True,
            truncation=True,
            return_tensors="tf"
        )

        predicted_embeddings = model(tokenized_docs)
        # print(tf_outputs)
        docs_dict = {}
        for doc_id, out in zip(doc_ids, predicted_embeddings[1]):
            docs_dict[doc_id] = out.numpy()

        return Vectorization.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=None,
                                                   return_vecs=return_vecs)

    @classmethod
    def longformer_tuned(cls, corpus: Corpus, save_path: str = "models/", return_vecs: bool = True):
        _, doc_ids = corpus.get_texts_and_doc_ids()

        model_name = "allenai/longformer-base-4096"  # "bert-base-uncased"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        documents = corpus.get_flat_documents()
        print(documents)
        tokenized_docs = tokenizer(
            documents,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # model = LongformerModel.from_pretrained('allenai/longformer-base-4096', return_dict=True)
        # tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        # tokenized = tokenizer(tokenized_docs, padding=True, truncation=True, return_tensors="pt")
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        model.train()

        # train_dataset = TokenizedDataset(tokenized)
        # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        optim = AdamW(model.parameters(), lr=5e-5)
        tokenized_batch = tokenized_docs
        nr_epochs = 2
        epoch_bar = tqdm(range(nr_epochs), desc=f'Epoch {0} with loss UNKKOWN')
        outputs = None
        for epoch in epoch_bar:
            # for tokenized_batch in train_loader:
            optim.zero_grad()
            tokenized_b = tokenized_batch.to(device)
            # input_ids = tokenized['input_ids'].to(device)
            # attention_mask = tokenized['attention_mask'].to(device)
            # labels = tokenized['input_ids'].to(device)
            # outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
            # loss = outputs.pooler_output
            outputs = model(**tokenized_b)
            loss = outputs[0]
            epoch_bar.set_description(desc=f'Epoch {epoch} with loss {loss.sum():.2f}')
            epoch_bar.update()
            loss.sum().backward()
            optim.step()

        docs_dict = {}
        for doc_id, out in zip(doc_ids, outputs[1]):
            docs_dict[doc_id] = out.detach().numpy()

        return Vectorization.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=None,
                                                   return_vecs=return_vecs)

    @classmethod
    def flair(cls, corpus: Corpus, word_embedding_base: Union[str, None], document_embedding: str,
              save_path: str = "models/", return_vecs: bool = True, chunk_len: int = None,
              sentence_based: bool = None, sentence_nr: int = None, pretuned: bool = False):
        """

        :param pretuned:
        :param sentence_nr:
        :param sentence_based:
        :param chunk_len:
        :param return_vecs:
        :param save_path:
        :param corpus:
        :param word_embedding_base: - 'glove', (only en), - 'fasttext'
        :param document_embedding:  pool vs rnn for w2v mode - bert: 'bert',  - 'longformer' (only en) -
        'flair', 'stacked-flair',
        """

        if (word_embedding_base == "glove" or document_embedding == "longformer") and corpus.language == Language.DE:
            raise UserWarning(f'English embeddings / model called on german text!')
        if word_embedding_base == "fasttext":
            if corpus.language == Language.DE:
                word_embedding_base = 'de'
            else:
                word_embedding_base = 'en'
        if document_embedding == "bert":
            if corpus.language == Language.DE:
                document_embedding = 'bert-de'
        if document_embedding == "flair" or document_embedding == "stacked-flair":
            if corpus.language == Language.DE:
                document_embedding = f'{document_embedding}-de'

        if sentence_based:
            documents = FlairSentenceDocumentIterator(corpus, sentence_nr=sentence_nr)
        else:
            documents = FlairDocumentIterator(corpus, chunk_len=chunk_len)
        docs_dict = cls.flair_base(documents, word_embedding_base=word_embedding_base,
                                   document_embedding=document_embedding, chunk_len=chunk_len,
                                   sentence_based=sentence_based, pretuned=pretuned)

        return Vectorization.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=None,
                                                   return_vecs=return_vecs)

    @classmethod
    def book2vec_simple(cls, corpus: Corpus, save_path: str = "models/",
                        disable_aspects: List[str] = None, return_vecs: bool = True, without_training: bool = False,
                        chunk_len: int = None,
                        facets_of_chunks: bool = True,
                        window_size: int = 0,
                        concat: bool = False,
                        dimension: int = None,
                        pretrained: bool = False,
                        use_dictionary_lookup: bool = False
                        ):

        lemma = False
        lower = False

        if disable_aspects is None:
            disable_aspects = []

        if "cont" not in disable_aspects:
            topic_dict = TopicModeller.topic_modelling(corpus)
            # topic_dict, _ = TopicModeller.train_lda_mem_eff(corpus)
        else:
            topic_dict = None

        if "plot" not in disable_aspects:
            summary_dict = Summarizer.get_summary(corpus)
        else:
            summary_dict = None

        documents = CorpusTaggedFacetIterator(corpus, lemma=lemma, lower=lower, disable_aspects=disable_aspects,
                                              topic_dict=topic_dict, summary_dict=summary_dict, chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window=window_size,
                                              use_dictionary_lookup=use_dictionary_lookup)
        # print('Start training')
        logging.info("Start training")

        # for document in documents:
        #     print(document)
        model, words_dict, docs_dict = cls.doc2vec_base(documents, without_training, chunk_len=chunk_len,
                                                        dimension=dimension, language=corpus.language,
                                                        pretrained=pretrained)

        aspect_path = os.path.basename(save_path)
        write_doc_based_aspect_frequency_analyzis(documents.doc_aspects, save_name=aspect_path)
        write_doc_based_aspect_frequency_analyzis(documents.detailed_aspects, save_name=f'{aspect_path}_detailed')
        # write_aspect_frequency_analyzis(doc_aspects=doc_aspects, doc_ids=doc_ids, save_name=aspect_path)

        # print(docs_dict.keys())

        # concat_vecs = Vectorizer.combine_vectors_by_concat(docs_dict)
        # Vectorization.store_vecs_and_reload(save_path=f'{save_path}_con', docs_dict=concat_vecs, words_dict=None,
        #                                     return_vecs=False)
        #
        # pca_vecs = Vectorizer.pca_on_vectors(concat_vecs)
        # Vectorization.store_vecs_and_reload(save_path=f'{save_path}_pca', docs_dict=pca_vecs, words_dict=None,
        #                                     return_vecs=False)
        #
        # docs_dict = Vectorizer.combine_vectors_by_sum(docs_dict)

        docs_dict = Vectorization.combine_vectors(save_path=save_path, document_dictionary=docs_dict, dim_size=None)

        # print(path)

        return Vectorization.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=words_dict,
                                                   return_vecs=return_vecs)

    @staticmethod
    def build_simple_sentence_aspects(aspect: List[List[str]], corpus: Corpus):
        # print(aspect)
        sentence_aspect = [[[token.representation() for token in sentence.tokens
                             if token.representation() in aspect[i]]
                            for sentence in document.sentences]
                           for i, document in enumerate(corpus.documents.values())]
        # print(sentence_aspect)
        return sentence_aspect

    @classmethod
    def topiv_vecs(cls, corpus: Corpus, save_path: str = "models/", return_vecs: bool = True):
        _, _, topic_model, lda_corpus, doc_ids = TopicModeller.train_lda_mem_eff(corpus)
        docs_dict = {}
        words_dict = None
        # print(len(lda_corpus))
        # print(doc_ids)
        for i, doc_id in enumerate(doc_ids):
            doc = lda_corpus[i]
            docs_dict[doc_id] = np.array([score for (topic, score) in topic_model[doc][0]])

        return Vectorization.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=None,
                                                   return_vecs=return_vecs)

    @classmethod
    def book2vec_adv(cls, corpus: Corpus, save_path: str = "models/",
                     disable_aspects: List[str] = None, return_vecs: bool = True, algorithm="doc2vec",
                     without_training: bool = False,
                     chunk_len: int = None,
                     facets_of_chunks: bool = True,
                     window_size: int = 0,
                     dimension: int = None,
                     pretrained: bool = False,
                     pretuned: bool = False,
                     use_dictionary_lookup: bool = False):
        lemma = False
        lower = False

        if disable_aspects is None:
            disable_aspects = []
        # disable_aspects.extend(["cont", "plot"])
        if "cont" not in disable_aspects:
            topic_dict = TopicModeller.topic_modelling(corpus)
            # topic_dict, _ = TopicModeller.train_lda_mem_eff(corpus)
        else:
            topic_dict = None

        if "plot" not in disable_aspects:
            summary_dict = Summarizer.get_summary(corpus)
        else:
            summary_dict = None

        logging.info("Start training")
        if algorithm.lower() == "doc2vec" or algorithm.lower() == "d2v":
            documents = CorpusTaggedFacetIterator(corpus, lemma=lemma, lower=lower, disable_aspects=disable_aspects,
                                                  topic_dict=topic_dict, summary_dict=summary_dict, chunk_len=chunk_len,
                                                  window=window_size,
                                                  use_dictionary_lookup=use_dictionary_lookup)
            model, words_dict, docs_dict = cls.doc2vec_base(documents, without_training, chunk_len=chunk_len,
                                                            dimension=dimension, language=corpus.language,
                                                            pretrained=pretrained)
        elif algorithm.lower() == "avg_w2v" or algorithm.lower() == "w2v" or algorithm.lower() == "word2vec":
            preprocessed_sentences = CorpusSentenceIterator(corpus)
            documents = CorpusTaggedFacetIterator(corpus, lemma=lemma, lower=lower, disable_aspects=disable_aspects,
                                                  topic_dict=topic_dict, summary_dict=summary_dict, window=window_size,
                                                  use_dictionary_lookup=use_dictionary_lookup)
            aspect_doc_ids = [d.tags[0] for d in documents]
            model, words_dict, docs_dict = cls.word2vec_base(preprocessed_sentences, documents,
                                                             aspect_doc_ids, without_training,
                                                             dimension=dimension,
                                                             language=corpus.language,
                                                             pretrained=pretrained)
        elif algorithm.lower() == "transformer":
            # documents = FlairDocumentIterator(corpus)
            documents = FlairFacetIterator(corpus, lemma=lemma, lower=lower, disable_aspects=disable_aspects,
                                           topic_dict=topic_dict, summary_dict=summary_dict, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window=window_size,
                                           use_dictionary_lookup=use_dictionary_lookup)
            words_dict = None
            docs_dict = cls.flair_base(documents, word_embedding_base=None,
                                       document_embedding="bert-de", chunk_len=chunk_len, pretuned=pretuned)
        else:
            raise UserWarning(f"Not supported vectorization algorithm '{algorithm}'!")

        aspect_path = os.path.basename(save_path)
        write_doc_based_aspect_frequency_analyzis(documents.doc_aspects, save_name=aspect_path)
        write_doc_based_aspect_frequency_analyzis(documents.detailed_aspects, save_name=f'{aspect_path}_detailed')

        # concat_vecs = Vectorizer.combine_vectors_by_concat(docs_dict)
        # Vectorization.store_vecs_and_reload(save_path=f'{save_path}_con', docs_dict=concat_vecs, words_dict=None,
        #                                     return_vecs=False)
        #
        # pca_vecs = Vectorizer.pca_on_vectors(concat_vecs)
        # Vectorization.store_vecs_and_reload(save_path=f'{save_path}_pca', docs_dict=pca_vecs, words_dict=None,
        #                                     return_vecs=False)
        #
        # docs_dict = Vectorizer.combine_vectors_by_sum(docs_dict)

        docs_dict = Vectorization.combine_vectors(save_path=save_path, document_dictionary=docs_dict, dim_size=None)

        return Vectorization.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=words_dict,
                                                   return_vecs=return_vecs)

    # @classmethod
    # def book2vec_adv(cls, corpus: Corpus, save_path: str = "models/",
    #                  disable_aspects: List[str] = None, return_vecs: bool = True, algorithm="doc2vec",
    #                  without_training: bool = False, enable_mode: bool = False):
    #     lemma = False
    #     lower = False
    #
    #     if disable_aspects is None:
    #         disable_aspects = []
    #     # documents = [TaggedDocument(doc, [i])
    #     #              for i, doc in enumerate(Preprocesser.tokenize(corpus.get_texts_and_doc_ids()))]
    #     # documents = [TaggedDocument(Preprocesser.tokenize(document.text), [doc_id])
    #     #              for doc_id, document in corpus.documents.items()]
    #     lan_model = corpus.give_spacy_lan_model()
    #     # print('>', preprocessed_documents)
    #     _, doc_ids = corpus.get_texts_and_doc_ids()
    #     if corpus.document_entities is None:
    #         raise UserWarning("No Entities set!")
    #     document_entities = corpus.get_document_entities_representation()
    #     # reverted_entities = Utils.revert_dictionaries(document_entities)
    #     # print('>', reverted_entities)
    #     times, locations = Vectorizer.resolve_entities(document_entities)
    #     # print(len(times), times)
    #
    #     aspects = {}
    #     # sentence_aspects = {}
    #
    #     if (enable_mode and "time" in disable_aspects) or (not enable_mode and "time" not in disable_aspects):
    #         aspects['time'] = Preprocesser.structure_string_texts(times, lan_model, lemma=lemma, lower=lower)
    #
    #     if (enable_mode and "loc" in disable_aspects) or (not enable_mode and "loc" not in disable_aspects):
    #         aspects['loc'] = Preprocesser.structure_string_texts(locations, lan_model, lemma=lemma, lower=lower)
    #
    #     if (enable_mode and "raw" in disable_aspects) or (not enable_mode and "raw" not in disable_aspects):
    #         aspects['raw'] = corpus.get_flat_document_tokens(lemma=lemma, lower=lower)
    #
    #     if (enable_mode and "atm" in disable_aspects) or (not enable_mode and "atm" not in disable_aspects):
    #         aspects['atm'] = corpus.get_flat_and_filtered_document_tokens(lemma=lemma,
    #                                                                       lower=lower,
    #                                                                       pos=["ADJ", "ADV"])
    #
    #     if (enable_mode and "sty" in disable_aspects) or (not enable_mode and "sty" not in disable_aspects):
    #         aspects['sty'] = corpus.get_flat_and_filtered_document_tokens(lemma=lemma,
    #                                                                       lower=lower,
    #                                                                       focus_stopwords=True)
    #
    #     if (enable_mode and "cont" in disable_aspects) or (not enable_mode and "cont" not in disable_aspects):
    #         _, topic_list = TopicModeller.train_lda(corpus)
    #         aspects["cont"] = topic_list
    #
    #     if (enable_mode and "plot" in disable_aspects) or (not enable_mode and "plot" not in disable_aspects):
    #         aspects["plot"] = Summarizer.get_corpus_summary_sentence_list(corpus,
    #                                                                       lemma=lemma,
    #                                                                       lower=lower)
    #
    #     # print(aspects_old.keys(), disable_aspects)
    #     assert set(aspects.keys()).union(disable_aspects) == {'time', 'loc', 'raw', 'atm', 'sty', 'cont', 'plot'}
    #     aspect_path = os.path.basename(save_path)
    #     write_aspect_frequency_analyzis(aspects=aspects, doc_ids=doc_ids, save_name=aspect_path)
    #
    #     documents = []
    #     for aspect_name, aspect_documents in aspects.items():
    #         documents.extend([TaggedDocument(preprocessed_document_text, [f'{doc_id}_{aspect_name}'])
    #                           for preprocessed_document_text, doc_id in zip(aspect_documents, doc_ids)])
    #
    #     logging.info("Start training")
    #     if algorithm.lower() == "doc2vec" or algorithm.lower() == "d2v":
    #         model, words_dict, docs_dict = cls.doc2vec_base(documents, without_training)
    #     elif algorithm.lower() == "avg_w2v" or algorithm.lower() == "w2v" or algorithm.lower() == "word2vec":
    #         preprocessed_sentences = corpus.get_flat_corpus_sentences()
    #         aspect_doc_ids = [d.tags[0] for d in documents]
    #         model, words_dict, docs_dict = cls.word2vec_base(preprocessed_sentences, documents,
    #                                                          aspect_doc_ids, without_training)
    #     else:
    #         raise UserWarning(f"Not supported vectorization algorithm '{algorithm}'!")
    #
    #     docs_dict = Vectorizer.combine_vectors_by_sum(docs_dict)
    #
    #     return Vectorizer.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=words_dict,
    #                                             return_vecs=return_vecs)

    @classmethod
    def random_aspect2vec(cls, corpus: Corpus, save_path: str = "models/",
                          return_vecs: bool = True, algorithm="doc2vec",
                          without_training: bool = False, dimension: int = None,
                          pretrained: bool = False):
        def nr_to_roman(nr: int):
            if nr == 0:
                return 'I'
            elif nr == 1:
                return 'II'
            elif nr == 2:
                return 'III'
            elif nr == 3:
                return 'IV'
            elif nr == 4:
                return 'V'
            elif nr == 5:
                return 'VI'
            elif nr == 6:
                return 'VII'
            elif nr == 7:
                return 'VIII'
            elif nr == 8:
                return 'IX'
            elif nr == 9:
                return 'X'
            elif nr == 10:
                return 'XI'

        lemma = False
        lower = False
        _, doc_ids = corpus.get_texts_and_doc_ids()
        if corpus.document_entities is None:
            raise UserWarning("No Entities set!")

        aspects = {}
        prob = 0.2
        nr_aspects = 5
        aspects['raw'] = corpus.get_flat_document_tokens(lemma=lemma, lower=lower)

        for aspect_nr in range(0, nr_aspects):
            aspects[f'aspect{nr_to_roman(aspect_nr)}'] = corpus.get_flat_and_random_document_tokens(prop_to_keep=prob,
                                                                                                    seed=aspect_nr,
                                                                                                    lemma=lemma,
                                                                                                    lower=lower)
        documents = []
        for aspect_name, aspect_documents in aspects.items():
            documents.extend([TaggedDocument(preprocessed_document_text, [f'{doc_id}_{aspect_name}'])
                              for preprocessed_document_text, doc_id in zip(aspect_documents, doc_ids)])

        logging.info("Start training")
        if algorithm.lower() == "doc2vec" or algorithm.lower() == "d2v":
            model, words_dict, docs_dict = cls.doc2vec_base(documents, without_training, dimension=dimension,
                                                            language=corpus.language,
                                                            pretrained=pretrained)
        elif algorithm.lower() == "avg_w2v" or algorithm.lower() == "w2v" or algorithm.lower() == "word2vec":
            preprocessed_sentences = corpus.get_flat_corpus_sentences()
            aspect_doc_ids = [d.tags[0] for d in documents]
            model, words_dict, docs_dict = cls.word2vec_base(preprocessed_sentences, documents,
                                                             aspect_doc_ids, without_training, dimension=dimension,
                                                             language=corpus.language,
                                                             pretrained=pretrained)
        else:
            raise UserWarning(f"Not supported vectorization algorithm '{algorithm}'!")
        docs_dict = Vectorization.combine_vectors_by_sum(docs_dict)

        return Vectorization.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=words_dict,
                                                   return_vecs=return_vecs)

    @staticmethod
    def model2dict(model: Doc2Vec):
        words_dict = {word: model.wv[word] for word in model.wv.vocab}
        doc_dict = {doc_id: model.docvecs[doc_id] for doc_id in model.docvecs.doctags}
        # print(model.__index_to_doctag(i, self.offset2doctag, self.max_rawint))
        # print('>>', model.docvecs.vectors_docs)
        # print('>>', model.docvecs.max_rawint)
        return words_dict, doc_dict
