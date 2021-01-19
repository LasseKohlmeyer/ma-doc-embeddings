import logging
import os
from collections import defaultdict
from typing import Union, List, Dict


from gensim.models import KeyedVectors, Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import utils
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile, datapath
from numpy import float32 as real
from tqdm import tqdm
import torch
from transformers import TFAutoModel, AutoTokenizer, AdamW, AutoModel

from corpus_iterators import CorpusSentenceIterator, \
    CorpusDocumentIterator, CorpusTaggedDocumentIterator, CorpusTaggedFacetIterator, \
    write_doc_based_aspect_frequency_analyzis, FlairDocumentIterator, FlairFacetIterator
from doc2vec_structures import DocumentKeyedVectors
from flair_connector import FlairConnector
from text_summarisation import Summarizer
from topic_modelling import TopicModeller
from corpus_structure import Corpus, ConfigLoader, DataHandler, Language

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
    pretrained_emb_path = None  # config["embeddings"]["pretrained"]
    # "E:/embeddings/glove.6B.300d.txt" # "E:/embeddings/google300.txt"
    pretrained_emb = robust_vec_loading(pretrained_emb_path, binary=False)

    @staticmethod
    def build_vec_file_name(number_of_subparts: Union[int, str], size: Union[int, str], dataset: str, filter_mode: str,
                            vectorization_algorithm: str, fake_series: str) \
            -> str:
        sub_path = DataHandler.build_config_str(number_of_subparts, size, dataset, filter_mode,
                                                vectorization_algorithm, fake_series)
        return os.path.join(config["system_storage"]["models"], f'{sub_path}.model')

    @staticmethod
    def algorithm(input_str: str, corpus: Corpus, save_path: str = "models/",
                  return_vecs: bool = False, chunk_len: int = None):
        if "_o_" in input_str:
            return

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
            return Vectorizer.avg_wv2doc(corpus, save_path, return_vecs=return_vecs)
        elif input_str == "avg_wv2doc_restrict10000":
            return Vectorizer.avg_wv2doc(corpus, save_path, return_vecs=return_vecs, without_training=True,
                                         restrict_to=10000)
        elif input_str == "avg_wv2doc_untrained":
            return Vectorizer.avg_wv2doc(corpus, save_path, return_vecs=return_vecs, without_training=True)
        elif input_str == "doc2vec":
            return Vectorizer.doc2vec(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len)
        elif input_str == "doc2vec_untrained":
            return Vectorizer.doc2vec(corpus, save_path, return_vecs=return_vecs, without_training=True,
                                      chunk_len=chunk_len)
        # elif input_str == "longformer" or "longformer_untuned" or "untuned_longformer":
        #     return Vectorizer.longformer_untuned(corpus, save_path, return_vecs=return_vecs)
        # elif input_str == "longformer_tuned" or "tuned_longformer":
        #     return Vectorizer.longformer_tuned(corpus, save_path, return_vecs=return_vecs)
        elif input_str == "book2vec_simple" or input_str == "book2vec":
            return Vectorizer.book2vec_simple(corpus, save_path, return_vecs=return_vecs,
                                              disable_aspects=['plot', 'cont'], chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_wo_raw":
            return Vectorizer.book2vec_simple(corpus, save_path,
                                              disable_aspects=['raw'], return_vecs=return_vecs, chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_wo_loc":
            return Vectorizer.book2vec_simple(corpus, save_path,
                                              disable_aspects=['loc'], return_vecs=return_vecs, chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_wo_time":
            return Vectorizer.book2vec_simple(corpus, save_path,
                                              disable_aspects=['time'], return_vecs=return_vecs, chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_wo_sty":
            return Vectorizer.book2vec_simple(corpus, save_path,
                                              disable_aspects=['sty'], return_vecs=return_vecs, chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_wo_atm":
            return Vectorizer.book2vec_simple(corpus, save_path,
                                              disable_aspects=['atm'], return_vecs=return_vecs, chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_simple_untrained" or input_str == "book2vec_untrained":
            return Vectorizer.book2vec_simple(corpus, save_path, return_vecs=return_vecs,
                                              without_training=True, chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv":
            return Vectorizer.book2vec_adv(corpus, save_path, return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv_wo_raw":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['raw'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv_wo_loc":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['loc'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv_wo_time":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['time'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv_wo_sty":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['sty'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv_wo_atm":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['atm'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv_wo_plot":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['plot'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv_wo_cont":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['cont'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv_wo_raw":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['raw'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv_wo_loc":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['loc'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv_wo_time":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['time'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv_wo_sty":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['sty'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv_wo_atm":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['atm'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv_wo_plot":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['plot'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_adv_wo_cont":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['cont'], return_vecs=return_vecs, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
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
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "book2vec_bert":
            return Vectorizer.book2vec_adv(corpus, save_path,
                                           disable_aspects=['cont'],
                                           return_vecs=return_vecs,
                                           algorithm="transformer",
                                           chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window_size=window)
        elif input_str == "random_aspect2vec" or input_str == "random":
            return Vectorizer.random_aspect2vec(corpus, save_path, return_vecs=return_vecs,
                                                algorithm="doc2vec")
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
        elif input_str == "longformer":
            return Vectorizer.flair(corpus, None, "longformer", save_path, return_vecs=return_vecs, chunk_len=chunk_len)
        elif input_str == "flair":
            return Vectorizer.flair(corpus, None, "flair", save_path, return_vecs=return_vecs, chunk_len=chunk_len)
        elif input_str == "stacked_flair":
            return Vectorizer.flair(corpus, None, "stacked_flair", save_path, return_vecs=return_vecs,
                                    chunk_len=chunk_len)
        else:
            raise UserWarning(f"fUnknown input string {input_str}!")

    @classmethod
    def word2vec_base(cls,
                      preprocessed_sentences: Union[List[List[str]], CorpusSentenceIterator],
                      preprocessed_documents: Union[List[str], CorpusDocumentIterator, CorpusTaggedFacetIterator],
                      doc_ids,
                      without_training: bool,
                      restrict_to: int = None):
        if cls.pretrained_emb_path:
            model = cls.pretrained_emb

        else:
            # model = Word2Vec(preprocessed_sentences, size=cls.dim, window=cls.window, min_count=cls.min_count,
            #                  workers=cls.workers, iter=cls.epochs, seed=cls.seed)
            model = Word2Vec(size=cls.dim, window=cls.window, min_count=cls.min_count,
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
                     without_training: bool, chunk_len: int = None):
        # model = Doc2Vec(documents, vector_size=100, window=10, min_count=2, workers=4, epochs=20)
        # model = Doc2Vec(documents, vector_size=cls.dim, window=cls.window, min_count=cls.min_count,
        #                 workers=cls.workers, epochs=cls.epochs, pretrained_emb=cls.pretrained_emb_path, seed=cls.seed)
        # print('train')
        model = Doc2Vec(vector_size=cls.dim, min_count=cls.min_count, epochs=cls.epochs,
                        pretrained_emb=cls.pretrained_emb_path, seed=cls.seed, workers=cls.workers,
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

        return model, words_dict, docs_dict

    @classmethod
    def flair_base(cls, documents: Union[List[str], FlairFacetIterator, FlairDocumentIterator],
                   word_embedding_base: str = None, document_embedding: str = None, chunk_len: int = None):
        """

        :param chunk_len: length of chunks
        :param documents: input documents
        :param word_embedding_base: - glove: 'glove', (only en), - fasttext: 'en', 'de'
        :param document_embedding:  pool vs rnn for w2v mode - bert: 'bert', 'bert-de'  - 'longformer' (only en) -
        'flair', 'stacked-flair', 'flair-de', 'stacked-flair-de'
        """
        flair_instance = FlairConnector(word_embedding_base=word_embedding_base, document_embedding=document_embedding)

        docs_dict = flair_instance.embedd_documents(documents)

        if chunk_len:
            docs_dict.update(cls.avg_sim_prefix_doc_ids(docs_dict))

        return docs_dict

    @classmethod
    def avg_wv2doc(cls, corpus: Corpus, save_path: str = "models/", return_vecs: bool = True,
                   without_training: bool = False, restrict_to: int = None):
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
                                                         restrict_to)

        return Vectorizer.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=words_dict,
                                                return_vecs=return_vecs)

    @classmethod
    def doc2vec(cls, corpus: Corpus, save_path: str = "models/", return_vecs: bool = True,
                without_training: bool = False, chunk_len: int = None):
        # documents = [TaggedDocument(doc, [i])
        #              for i, doc in enumerate(Preprocesser.tokenize(corpus.get_texts_and_doc_ids()))]
        # documents = [TaggedDocument(Preprocesser.tokenize(document.text), [doc_id])
        #              for doc_id, document in corpus.documents.items()]

        documents = CorpusTaggedDocumentIterator(corpus, chunk_len=chunk_len)

        model, words_dict, docs_dict = cls.doc2vec_base(documents, without_training, chunk_len=chunk_len)

        return Vectorizer.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=words_dict,
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

        return Vectorizer.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=None,
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

        return Vectorizer.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=None,
                                                return_vecs=return_vecs)

    @classmethod
    def flair(cls, corpus: Corpus, word_embedding_base: Union[str, None], document_embedding: str,
              save_path: str = "models/", return_vecs: bool = True, chunk_len: int = None):
        """

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

        documents = FlairDocumentIterator(corpus, chunk_len=chunk_len)
        docs_dict = cls.flair_base(documents, word_embedding_base=word_embedding_base,
                                   document_embedding=document_embedding, chunk_len=chunk_len)

        return Vectorizer.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=None,
                                                return_vecs=return_vecs)

    @classmethod
    def book2vec_simple(cls, corpus: Corpus, save_path: str = "models/",
                        disable_aspects: List[str] = None, return_vecs: bool = True, without_training: bool = False,
                        chunk_len: int = None,
                        facets_of_chunks: bool = True,
                        window_size: int = 0):

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

            if corpus.root_corpus_path is None:
                raise UserWarning("No root corpus set!")
            summary_dict = Summarizer.get_summary(corpus.root_corpus_path)
        else:
            summary_dict = None

        documents = CorpusTaggedFacetIterator(corpus, lemma=lemma, lower=lower, disable_aspects=disable_aspects,
                                              topic_dict=topic_dict, summary_dict=summary_dict, chunk_len=chunk_len,
                                              facets_of_chunks=facets_of_chunks, window=window_size)
        # print('Start training')
        logging.info("Start training")

        # for document in documents:
        #     print(document)
        model, words_dict, docs_dict = cls.doc2vec_base(documents, without_training, chunk_len=chunk_len)

        aspect_path = os.path.basename(save_path)
        write_doc_based_aspect_frequency_analyzis(documents.doc_aspects, save_name=aspect_path)
        # write_aspect_frequency_analyzis(doc_aspects=doc_aspects, doc_ids=doc_ids, save_name=aspect_path)

        # print(docs_dict.keys())
        docs_dict = Vectorizer.combine_vectors_by_sum(docs_dict)
        # print(path)

        return Vectorizer.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=words_dict,
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
    def book2vec_adv(cls, corpus: Corpus, save_path: str = "models/",
                     disable_aspects: List[str] = None, return_vecs: bool = True, algorithm="doc2vec",
                     without_training: bool = False,
                     chunk_len: int = None,
                     facets_of_chunks: bool = True,
                     window_size: int = 0):
        lemma = False
        lower = False

        if disable_aspects is None:
            disable_aspects = []
        disable_aspects.extend(["cont", "plot"])
        if "cont" not in disable_aspects:
            topic_dict = TopicModeller.topic_modelling(corpus)
            # topic_dict, _ = TopicModeller.train_lda_mem_eff(corpus)
        else:
            topic_dict = None

        if "plot" not in disable_aspects:
            if corpus.root_corpus_path is None:
                raise UserWarning("No root corpus set!")
            summary_dict = Summarizer.get_summary(corpus.root_corpus_path)
        else:
            summary_dict = None

        logging.info("Start training")
        if algorithm.lower() == "doc2vec" or algorithm.lower() == "d2v":
            documents = CorpusTaggedFacetIterator(corpus, lemma=lemma, lower=lower, disable_aspects=disable_aspects,
                                                  topic_dict=topic_dict, summary_dict=summary_dict, chunk_len=chunk_len)
            model, words_dict, docs_dict = cls.doc2vec_base(documents, without_training, chunk_len=chunk_len)
        elif algorithm.lower() == "avg_w2v" or algorithm.lower() == "w2v" or algorithm.lower() == "word2vec":
            preprocessed_sentences = CorpusSentenceIterator(corpus)
            documents = CorpusTaggedFacetIterator(corpus, lemma=lemma, lower=lower, disable_aspects=disable_aspects,
                                                  topic_dict=topic_dict, summary_dict=summary_dict, window=window_size)
            aspect_doc_ids = [d.tags[0] for d in documents]
            model, words_dict, docs_dict = cls.word2vec_base(preprocessed_sentences, documents,
                                                             aspect_doc_ids, without_training)
        elif algorithm.lower() == "transformer":
            # documents = FlairDocumentIterator(corpus)
            documents = FlairFacetIterator(corpus, lemma=lemma, lower=lower, disable_aspects=disable_aspects,
                                           topic_dict=topic_dict, summary_dict=summary_dict, chunk_len=chunk_len,
                                           facets_of_chunks=facets_of_chunks, window=window_size)
            words_dict = None
            docs_dict = cls.flair_base(documents, word_embedding_base=None,
                                       document_embedding="bert-de", chunk_len=chunk_len)
        else:
            raise UserWarning(f"Not supported vectorization algorithm '{algorithm}'!")

        aspect_path = os.path.basename(save_path)
        write_doc_based_aspect_frequency_analyzis(documents.doc_aspects, save_name=aspect_path)

        docs_dict = Vectorizer.combine_vectors_by_sum(docs_dict)

        return Vectorizer.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=words_dict,
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
                          without_training: bool = False):
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
            model, words_dict, docs_dict = cls.doc2vec_base(documents, without_training)
        elif algorithm.lower() == "avg_w2v" or algorithm.lower() == "w2v" or algorithm.lower() == "word2vec":
            preprocessed_sentences = corpus.get_flat_corpus_sentences()
            aspect_doc_ids = [d.tags[0] for d in documents]
            model, words_dict, docs_dict = cls.word2vec_base(preprocessed_sentences, documents,
                                                             aspect_doc_ids, without_training)
        else:
            raise UserWarning(f"Not supported vectorization algorithm '{algorithm}'!")
        docs_dict = Vectorizer.combine_vectors_by_sum(docs_dict)

        return Vectorizer.store_vecs_and_reload(save_path=save_path, docs_dict=docs_dict, words_dict=words_dict,
                                                return_vecs=return_vecs)

    @staticmethod
    def combine_vectors_by_sum(document_dictionary: Dict[str, np.array]):
        summed_vecs = {}
        # print(document_dictionary.keys())

        if list(document_dictionary.keys())[0][-1].isdigit():
            element_pointer = -2
        else:
            element_pointer = -1
        base_ending_candidates = set([f"_{tag.split('_')[element_pointer]}" for tag in document_dictionary.keys()])
        print(base_ending_candidates, document_dictionary.keys())
        candidate_counter_dict = defaultdict(int)
        plain_doc_ids = set()
        for base_ending_candidate in base_ending_candidates:
            for doc_id in document_dictionary.keys():
                splitted_id = doc_id.split('_')
                if doc_id[-1].isdigit():
                    pass
                else:
                    prefix = '_'.join(splitted_id[:-1])
                    suffix = f"_{splitted_id[-1]}"
                    plain_doc_ids.add(prefix)

                    if base_ending_candidate == suffix:
                        candidate_counter_dict[base_ending_candidate] += 1
        print(len(plain_doc_ids), candidate_counter_dict)
        final_candidates = [candidate for candidate, count in candidate_counter_dict.items()
                            if count == len(plain_doc_ids)]

        if len(final_candidates) == 0:
            raise UserWarning("No aspect found for all documents")
        base_ending = final_candidates[0]

        id_groups = set([tag.split('_')[-1] for tag in document_dictionary.keys() if not tag.endswith(base_ending)])
        # print(base_ending, id_groups, document_dictionary.keys())

        for tag in document_dictionary.keys():
            if tag.endswith(base_ending):
                new_vec = document_dictionary[tag]
                base_tag = tag.replace(base_ending, '')
                for group in id_groups:
                    try:
                        new_vec += document_dictionary[f'{base_tag}_{group}']
                    except KeyError:
                        pass
                summed_vecs[f'{base_tag}'] = new_vec
            # print(model.docvecs[tag])
        summed_vecs.update(document_dictionary)
        return summed_vecs

    @staticmethod
    def model2dict(model: Doc2Vec):
        words_dict = {word: model.wv[word] for word in model.wv.vocab}
        doc_dict = {doc_id: model.docvecs[doc_id] for doc_id in model.docvecs.doctags}
        # print(model.__index_to_doctag(i, self.offset2doctag, self.max_rawint))
        # print('>>', model.docvecs.vectors_docs)
        # print('>>', model.docvecs.max_rawint)
        return words_dict, doc_dict

    @staticmethod
    def my_save_word2vec_format(fname: str, vocab: Dict[str, np.ndarray], vectors: np.ndarray, binary: bool = True,
                                total_vec: int = 2):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.
        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        vocab : dict
            The vocabulary of words.
        vectors : numpy.array
            The vectors to be stored.
        binary : bool, optional
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec : int, optional
            Explicitly specify total number of vectors
            (in case word vectors are appended with document vectors afterwards).
        """
        if not (vocab or vectors):
            raise RuntimeError("no input")
        if total_vec is None:
            total_vec = len(vocab)
        vector_size = vectors.shape[1]
        assert (len(vocab), vector_size) == vectors.shape

        with utils.open(fname, 'wb') as fout:
            fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
            # store in sorted order: most frequent words at the top
            for word, row in vocab.items():
                if binary:
                    row = row.astype(real)
                    # noinspection PyTypeChecker
                    fout.write(utils.to_utf8(word) + b" " + row.tostring())
                else:
                    fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))

    @staticmethod
    def my_save_docvec_word2vec_format(fname, docvecs: Dict[str, np.ndarray], prefix='*dt_', fvocab=None,
                                       total_vec=None, binary=False, write_first_line=True):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        docvecs : Dict[str, np.array]: actual document vectors
        prefix : str, optional
            Uniquely identifies doctags from word vocab, and avoids collision
            in case of repeated string in doctag and word vocab.
        fvocab : str, optional
            UNUSED.
        total_vec : int, optional
            Explicitly specify total no. of vectors
            (in case word vectors are appended with document vectors afterwards)
        binary : bool, optional
            If True, the data will be saved in binary word2vec format, else it will be saved in plain text.
        write_first_line : bool, optional
            Whether to print the first line in the file. Useful when saving doc-vectors after word-vectors.


        """

        def _index_to_doctag(i_index, offset, max_rawint_value):
            """Get string key for given `i_index`, if available. Otherwise return raw int doctag (same int)."""
            candidate_offset = i_index - max_rawint_value - 1
            if 0 <= candidate_offset < len(offset):
                return offset[candidate_offset]
            else:
                return i_index

        len_docvecs = len(docvecs)
        offset2doctag = list(docvecs.keys())
        vectors_docs = np.array(list(docvecs.values()))

        max_rawint = -1

        total_vec = total_vec or len_docvecs
        with utils.open(fname, 'ab') as fout:
            if write_first_line:
                fout.write(utils.to_utf8("%s %s\n" % (total_vec, vectors_docs.shape[1])))
            # store as in input order
            for i in range(len_docvecs):
                doctag = u"%s%s" % (prefix, _index_to_doctag(i, offset2doctag, max_rawint))
                row = vectors_docs[i]
                if binary:
                    # noinspection PyTypeChecker
                    fout.write(utils.to_utf8(doctag) + b" " + row.tostring())
                else:
                    fout.write(utils.to_utf8("%s %s\n" % (doctag, ' '.join("%f" % val for val in row))))

    @staticmethod
    def my_save_doc2vec_format(fname, doctag_vec: Dict[str, np.ndarray] = None, word_vec: Dict[str, np.ndarray] = None,
                               prefix='*dt_', fvocab=None, binary=False):
        """Store the input-hidden weight matrix in the same format used by the original C word2vec-tool.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        doctag_vec : bool, optional
            Indicates whether to store document vectors.
        word_vec : bool, optional
            Indicates whether to store word vectors.
        prefix : str, optional
            Uniquely identifies doctags from word vocab, and avoids collision in case of repeated string in doctag
            and word vocab.
        fvocab : str, optional
            Optional file path used to save the vocabulary.
        binary : bool, optional
            If True, the data will be saved in binary word2vec format, otherwise - will be saved in plain text.

        """
        if os.path.isfile(fname):
            os.remove(fname)
        docvecs = doctag_vec
        wv_vocab = word_vec  # self.wv.vocab

        write_first_line = False
        # save word vectors
        if word_vec:
            if not doctag_vec:
                total_vec = len(wv_vocab)
            else:
                total_vec = len(wv_vocab) + len(docvecs)
            wv_vectors = np.array(list(word_vec.values()))  # self.wv.vectors
            # self.wv.save_word2vec_format(fname, fvocab, binary, total_vec)
            Vectorizer.my_save_word2vec_format(fname, wv_vocab, wv_vectors, binary, total_vec)
        # save document vectors
        if doctag_vec:
            if not word_vec:
                total_vec = len(docvecs)
                write_first_line = True
            else:
                total_vec = len(wv_vocab) + len(docvecs)
            # self.docvecs.save_word2vec_format(
            #     fname, prefix=prefix, fvocab=fvocab, total_vec=total_vec,
            #     binary=binary, write_first_line=write_first_line)
            Vectorizer.my_save_docvec_word2vec_format(fname, doctag_vec, prefix, fvocab, total_vec, binary,
                                                      write_first_line)

    @staticmethod
    def my_load_doc2vec_format(fname: str, binary: bool = False):
        return DocumentKeyedVectors(KeyedVectors.load_word2vec_format(fname=fname, binary=binary))

    # @staticmethod
    # def show_results(model: Union[Doc2Vec, DocumentKeyedVectors], corpus: Corpus):
    #
    #     # print(model.docvecs[0])
    #     # print(model.docvecs.doctags)
    #     # print(model.docvecs.distance())
    #
    #     # print(model.wv.most_similar('God'))
    #     # print(model.wv.most_similar([model['God']]))
    #     # print('------')
    #     # print(model.wv.most_similar([model.docvecs['bs_0']]))
    #     # print('--')
    #     Vectorizer.most_similar_words_to_documents(model, positives=['bs_0'])
    #
    #     # finish = 100
    #     # c = 0
    #     # for doc_id, document in corpus.documents.items():
    #     #     print(doc_id, corpus.id2desc(doc_id), model.wv.most_similar([model.docvecs[doc_id]]))
    #     #     if c > finish:
    #     #         break
    #     #     c += 1
    #     # print('------')
    #     # print(model.wv.most_similar(positive=[model.docvecs['bs_0'], model.docvecs['bs_1']],
    #     #                             negative=[model.docvecs['bs_2']]))
    #     # print('--')
    #     Vectorizer.most_similar_words_to_documents(model, positives=['bs_0', 'bs_1'], negatives=['bs_2'])
    #     # print('------')
    #     # for result in model.docvecs.most_similar(positive=[model.docvecs['bs_0'], model.docvecs['bs_1']],
    #     #                                          negative=[model.docvecs['bs_2']]):
    #     #     index, sim = result
    #     #     print(index, corpus.id2desc(index), sim)
    #     # print('--')
    #     Vectorizer.most_similar_documents_to_documents(model, corpus, positives=['bs_0', 'bs_1'], negatives=['bs_2'])
    #
    #     # print('------')
    #     # for result in model.docvecs.most_similar([model['God']]):
    #     #     index, sim = result
    #     #     print(index, corpus.id2desc(index), sim)
    #     # print('--')
    #     Vectorizer.most_similar_documents_to_words(model, corpus, positives=['God'])
    #
    #     # print('------')
    #     # for result in model.docvecs.most_similar(positive=[model['woman'], model['king']], negative=[model['man']]):
    #     #     index, sim = result
    #     #     print(index, corpus.id2desc(index), sim)
    #     # print('--')
    #     Vectorizer.most_similar_documents_to_words(model, corpus, positives=['woman', 'god'], negatives=['man'])
    #     # Vectorizer.most_similar_documents_to_words(model, corpus, positives=['queen'])

    @staticmethod
    def get_topn_of_same_type_recursively(model: Union[Doc2Vec, DocumentKeyedVectors],
                                          positive_tags: List[str],
                                          positive_list: List[str], negative_list: List[str],
                                          topn: int,
                                          feature_to_use: str = None,
                                          origin_topn: int = None):

        def extract_feature(tag_list):
            tag = tag_list[0]
            if tag[-1].isdigit():
                return 'NF'
            splitt = tag.split('_')
            # print(splitt[-1])
            return f"_{splitt[-1]}"

        high_topn = topn * 10
        # print(feature_to_use)
        if feature_to_use:
            feature = feature_to_use
        else:
            feature = extract_feature(positive_tags)
        results = model.docvecs.most_similar(positive=positive_list, negative=negative_list, topn=high_topn)

        if feature == 'NF':
            results = [result for result in results if result[0][-1].isdigit()]
        else:
            results = [result for result in results if result[0].endswith(feature)]

        # print(results)
        if origin_topn is None:
            origin_topn = topn

        if len(results) >= origin_topn:
            return results[:origin_topn]
        else:
            return Vectorizer.get_topn_of_same_type_recursively(model, positive_tags, positive_list, negative_list,
                                                                high_topn,
                                                                feature_to_use, origin_topn)[:topn]

    @staticmethod
    def doctag_filter(tag: str, series: bool = False):
        splitted_tag = tag.split('_')
        digit_counter = 0
        for part in splitted_tag:
            if part[-1].isdigit():
                digit_counter += 1
        if series:
            limit = 3
        else:
            limit = 2
        if digit_counter >= limit:
            return False
        else:
            return True

    @staticmethod
    def get_ordered_results_of_same_type(model: Union[Doc2Vec, DocumentKeyedVectors],
                                         positive_tags: List[str],
                                         positive_list: List[str], negative_list: List[str],
                                         feature_to_use: str = None,
                                         series: bool = False):

        def extract_feature(tag_list):
            tag = tag_list[0]
            if tag[-1].isdigit():
                return 'NF'
            splitt = tag.split('_')
            # print(splitt[-1])
            return f"_{splitt[-1]}"

        if feature_to_use:
            feature = feature_to_use
        else:
            feature = extract_feature(positive_tags)
        # print(model.docvecs.doctags)
        # print(positive_list)

        results = model.docvecs.most_similar(positive=positive_list, negative=negative_list,
                                             topn=len(model.docvecs.doctags))
        alpha_ends = set([str(doc_id)[-1].isalpha() for doc_id, sim in results])
        if not any(alpha_ends):
            feature = "NF"

        # print(feature)
        if feature == 'NF':
            results = [result for result in results if result[0][-1].isdigit()]
        else:
            results = [result for result in results if result[0].endswith(feature)]

        results = [result for result in results if Vectorizer.doctag_filter(result[0], series)]

        return results

    # @staticmethod
    # def most_similar_words_to_documents(model: Union[Doc2Vec, DocumentKeyedVectors],
    #                                     positives: List[str],
    #                                     negatives: List[str] = None,
    #                                     topn: int = 10):
    #     if negatives is None:
    #         negatives = []
    #     positive_list = []
    #     for word in positives:
    #         positive_list.append(model.docvecs[word])
    #
    #     negative_list = []
    #     for word in negatives:
    #         negative_list.append(model.docvecs[word])
    #     results = model.wv.most_similar(positive=positive_list, negative=negative_list, topn=topn)
    #     for result in results:
    #         word, sim = result
    #         print(word, sim)
    #
    # @staticmethod
    # def most_similar_documents_to_words(model: Union[Doc2Vec, DocumentKeyedVectors],
    #                                     corpus: Corpus, positives: List[str],
    #                                     negatives=None,
    #                                     topn: int = 10,
    #                                     restrict_to_same: bool = True,
    #                                     feature_to_use: str = None):
    #     if feature_to_use is None:
    #         feature_to_use = "NF"
    #     if negatives is None:
    #         negatives = []
    #     positive_list = []
    #     for word in positives:
    #         positive_list.append(model.wv[word])
    #
    #     negative_list = []
    #     for word in negatives:
    #         negative_list.append(model.wv[word])
    #
    #    if restrict_to_same:
    #        results = Vectorizer.get_topn_of_same_type_recursively(model, positives, positive_list,
    #                                                               negative_list, topn,
    #                                                               feature_to_use=feature_to_use)
    #     else:
    #         results = model.docvecs.most_similar(positive=positive_list, negative=negative_list, topn=topn)
    #     for result in results:
    #         index, sim = result
    #         print(index, corpus.id2desc(index), sim)
    #
    # @staticmethod
    # def most_similar_documents_to_documents(model: Union[Doc2Vec, DocumentKeyedVectors],
    #                                         corpus: Corpus, positives: Union[List[str], str],
    #                                         negatives: Union[List[str], str] = None,
    #                                         topn: int = 10,
    #                                         restrict_to_same: bool = True,
    #                                         feature_to_use: str = None):
    #     positive_list = []
    #     if isinstance(positives, str):
    #         positives = [positives]
    #
    #     for doc_id in positives:
    #         positive_list.append(model.docvecs[doc_id])
    #
    #     if negatives is None:
    #         negatives = []
    #     elif isinstance(negatives, str):
    #         negatives = [negatives]
    #     else:
    #         pass
    #
    #     negative_list = []
    #     for doc_id in negatives:
    #         negative_list.append(model.docvecs[doc_id])
    #
    #     if restrict_to_same:
    #         results = Vectorizer.get_topn_of_same_type_recursively(model, positives, positive_list,
    #                                                                negative_list, topn,
    #                                                                feature_to_use)
    #     else:
    #         results = model.docvecs.most_similar(positive=positive_list, negative=negative_list, topn=topn)
    #     for result in results:
    #         index, sim = result
    #         print(index, corpus.id2desc(index), sim)

    @staticmethod
    def get_facet_sims(model: Union[Doc2Vec, DocumentKeyedVectors],
                       corpus: Corpus,
                       id_a: str, id_b: str,
                       print_results: bool = False):
        def build_facet_dict(facet_ids):
            facet_dict = {}
            for doc_id in facet_ids:
                if str(doc_id)[-1].isalpha():
                    facet_ind = doc_id.split('_')[-1]
                    facet_dict[facet_ind] = doc_id
                else:
                    facet_dict['sum'] = doc_id
            return facet_dict
        a_facet_ids = []
        b_facet_ids = []
        for doctag in model.docvecs.doctags:
            if str(doctag).startswith(id_a):
                a_facet_ids.append(doctag)

            if str(doctag).startswith(id_b):
                b_facet_ids.append(doctag)

        a_facets = build_facet_dict(a_facet_ids)
        b_facets = build_facet_dict(b_facet_ids)
        if a_facets.keys() != b_facets.keys():
            raise UserWarning("Found different facets!")

        similarity_tuples = []
        for facet in a_facets.keys():
            if print_results:
                print(facet, id_a, id_b, model.docvecs.similarity(a_facets[facet], b_facets[facet]))
            similarity_tuples.append((facet, f'{id_a} {corpus.documents[id_a].title}',
                                             f'{id_b} {corpus.documents[id_b].title}',
                                      model.docvecs.similarity(a_facets[facet], b_facets[facet])))

        return similarity_tuples

    @staticmethod
    def vector(model_vectors: Union[Doc2Vec, DocumentKeyedVectors], doc_id: str, facet_name: str,
               facet_mapping: Dict[str, str] = None):
        if facet_mapping:
            facet_name = facet_mapping[facet_name]

        doctag = f'{doc_id}_{facet_name}'

        return model_vectors.docvecs[doctag]

    @staticmethod
    def facet_sim(model_vectors: Union[Doc2Vec, DocumentKeyedVectors], doc_id_a: str, doc_id_b: str, facet_name: str,
                  facet_mapping: Dict[str, str] = None):
        if facet_mapping:
            facet_name = facet_mapping[facet_name]

        if facet_name == "":
            doctag_a = doc_id_a
            doctag_b = doc_id_b
        else:
            doctag_a = f'{doc_id_a}_{facet_name}'
            doctag_b = f'{doc_id_b}_{facet_name}'

        if doctag_a not in model_vectors.docvecs.doctags or doctag_b not in model_vectors.docvecs.doctags:
            doctag_a = doc_id_a
            doctag_b = doc_id_b

        return model_vectors.docvecs.similarity(doctag_a, doctag_b)

    @staticmethod
    def get_list(input_list: List[str], input_model: Union[Doc2Vec, DocumentKeyedVectors],
                 feature_to_use: Union[str, None]):
        out_list = []
        if input_list is None:
            input_list = []
        elif isinstance(input_list, str):
            input_list = [input_list]
        else:
            pass

        if feature_to_use and feature_to_use != "NF":
            input_list = [f'{element}_{feature_to_use}' for element in input_list]

        for document_id in input_list:
            try:
                out_list.append(input_model.docvecs[document_id])
            except KeyError:
                try:
                    out_list.append(input_model.wv[document_id])
                except KeyError:
                    try:
                        if document_id.count('_') > 1:
                            nr = '_'.join(document_id.split('_')[1:])
                        else:
                            nr = document_id
                        prefix = list(input_model.docvecs.doctags.keys())[0].split('_')[0]
                        new_id = f'{prefix}_{nr}'
                        # print(new_id)
                        out_list.append(input_model.docvecs[new_id])
                    except KeyError:
                        new_id = '_'.join(document_id.split('_')[:-1])
                        out_list.append(input_model.docvecs[new_id])

        return out_list

    @staticmethod
    def most_similar_documents(model: Union[Doc2Vec, DocumentKeyedVectors],
                               corpus: Corpus, positives: Union[List[str], str],
                               negatives: Union[List[str], str] = None,
                               topn: int = 10,
                               restrict_to_same: bool = True,
                               feature_to_use: str = None,
                               print_results: bool = True,
                               series: bool = False):

        positive_list = Vectorizer.get_list(positives, model, feature_to_use)
        negative_list = Vectorizer.get_list(negatives, model, feature_to_use)

        if restrict_to_same:
            results = Vectorizer.get_ordered_results_of_same_type(model, positives, positive_list, negative_list,
                                                                  feature_to_use, series)
        else:
            results = model.docvecs.most_similar(positive=positive_list, negative=negative_list,
                                                 topn=len(model.docvecs.doctags))
        results = [result for result in results if corpus.vector_doc_id_base_in_corpus(result[0])]
        results = results[:topn]

        if print_results:
            for result in results:
                index, sim = result
                print(index, corpus.id2desc(index), sim)
        return results

    @staticmethod
    def most_similar_words(model: Union[Doc2Vec, DocumentKeyedVectors],
                           positives: List[str],
                           negatives: List[str] = None,
                           topn: int = 10,
                           feature_to_use: str = None,
                           print_results: bool = True):
        # if negatives is None:
        #     negatives = []
        # positive_list = []
        # for word in positives:
        #     try:
        #         positive_list.append(model.docvecs[word])
        #     except KeyError:
        #         positive_list.append(model.wv[word])
        #
        # negative_list = []
        # for word in negatives:
        #     try:
        #         negative_list.append(model.docvecs[word])
        #     except KeyError:
        #         negative_list.append(model.wv[word])

        positive_list = Vectorizer.get_list(positives, model, feature_to_use=feature_to_use)
        negative_list = Vectorizer.get_list(negatives, model, feature_to_use=feature_to_use)

        results = model.wv.most_similar(positive=positive_list, negative=negative_list, topn=topn)

        if print_results:
            for result in results:
                word, sim = result
                print(word, sim)
        return results

    @staticmethod
    def store_vecs_and_reload(save_path: str,
                              docs_dict: Dict,
                              words_dict: Union[None, Dict],
                              return_vecs: bool,
                              binary: bool = False):
        Vectorizer.my_save_doc2vec_format(fname=save_path,
                                          doctag_vec=docs_dict,
                                          word_vec=words_dict,
                                          prefix='*dt_',
                                          fvocab=None,
                                          binary=binary)

        if return_vecs:
            vecs = Vectorizer.my_load_doc2vec_format(fname=save_path, binary=binary)
            return vecs
        else:
            return True
