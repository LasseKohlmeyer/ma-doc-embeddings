from typing import Union, List, Dict, Tuple

# import gensim
import spacy
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import utils
import numpy as np
from numpy import float32 as real

from doc2vec_structures import DocumentKeyedVectors
from utils import Preprocesser, Corpus, Utils, Language


class Vectorizer:
    @staticmethod
    def avg_wv2doc(corpus: Corpus, lemmatize: bool = False, lower: bool = False,
                   pos_filter: list = None, remove_stopwords: bool = False,
                   remove_punctuation: bool = True,
                   ner: bool = True,
                   save_path: str = "models/my_model"):
        texts, doc_ids = corpus.get_texts_and_doc_ids()
        lan_model = corpus.give_spacy_lan_model()
        preprocessed_sentences, _, preprocessed_documents = Preprocesser.preprocess(texts, lemmatize, lower, pos_filter,
                                                                                    remove_stopwords,
                                                                                    remove_punctuation, lan_model, ner,
                                                                                    return_in_sentence_format=True)
        # print('sents', preprocessed_sentences)
        # print(preprocessed_documents)
        print(f'Vectorize {len(preprocessed_sentences)} sentences and {len(preprocessed_documents)} documents')
        model = Word2Vec(preprocessed_sentences, size=100, window=5, min_count=1, workers=4, iter=20)
        docs_dict = {}
        for doc_id, doc in zip(doc_ids, preprocessed_documents):
            vector = []
            # print(doc)

            for token in doc:
                # print(token, model.wv.vocab[token])
                vector.append(model.wv[token])
            vector = sum(np.array(vector)) / len(doc)
            docs_dict[doc_id] = vector

        path = f"{save_path}_avg_wv2doc.model"
        words_dict = {word: model.wv[word] for word in model.wv.vocab}
        Vectorizer.my_save_doc2vec_format(fname=path, doctag_vec=docs_dict, word_vec=words_dict,
                                          prefix='*dt_',
                                          fvocab=None, binary=False)

        vecs = Vectorizer.my_load_doc2vec_format(fname=path, binary=False)
        return vecs

    @staticmethod
    def doc2vec(corpus: Corpus, lemmatize: bool = False, lower: bool = False,
                pos_filter: list = None, remove_stopwords: bool = False,
                remove_punctuation: bool = True,
                ner: bool = True,
                save_path: str = "models/my_model"):
        # documents = [TaggedDocument(doc, [i])
        #              for i, doc in enumerate(Preprocesser.tokenize(corpus.get_texts_and_doc_ids()))]
        # documents = [TaggedDocument(Preprocesser.tokenize(document.text), [doc_id])
        #              for doc_id, document in corpus.documents.items()]
        texts, doc_ids = corpus.get_texts_and_doc_ids()
        lan_model = corpus.give_spacy_lan_model()
        preprocessed_documents, _, _ = Preprocesser.preprocess(texts, lemmatize, lower, pos_filter, remove_stopwords,
                                                               remove_punctuation, lan_model, ner)
        documents = [TaggedDocument(preprocessed_document_text, [doc_id])
                     for preprocessed_document_text, doc_id in zip(preprocessed_documents, doc_ids)]
        # print(documents[0])
        # model = Doc2Vec(documents, vector_size=100, window=10, min_count=2, workers=4, epochs=20)
        pretrained_emb = None  #"E:/embeddings"
        model = Doc2Vec(vector_size=100, min_count=2, epochs=20,
                        pretrained_emb=pretrained_emb)
        model.build_vocab(documents)

        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)


        path = f"{save_path}_doc2vec.model"
        words_dict, docs_dict = Vectorizer.model2dict(model)
        Vectorizer.my_save_doc2vec_format(fname=path, doctag_vec=docs_dict, word_vec=words_dict,
                                          prefix='*dt_',
                                          fvocab=None, binary=False)

        vecs = Vectorizer.my_load_doc2vec_format(fname=path, binary=False)
        return vecs

    @staticmethod
    def book2vec_risch(corpus: Corpus, lemmatize: bool = False, lower: bool = False,
                       pos_filter: list = None, remove_stopwords: bool = False,
                       remove_punctuation: bool = True,
                       ner: bool = True,
                       save_path: str = "models/my_model"):
        # documents = [TaggedDocument(doc, [i])
        #              for i, doc in enumerate(Preprocesser.tokenize(corpus.get_texts_and_doc_ids()))]
        # documents = [TaggedDocument(Preprocesser.tokenize(document.text), [doc_id])
        #              for doc_id, document in corpus.documents.items()]
        texts, doc_ids = corpus.get_texts_and_doc_ids()
        lan_model = corpus.give_spacy_lan_model()
        preprocessed_documents, entities_of_documents, _ = Preprocesser.preprocess(texts, lemmatize, lower, pos_filter,
                                                                                   remove_stopwords,
                                                                                   remove_punctuation, lan_model, ner)
        # print('>', preprocessed_documents)

        reverted_entities = Utils.revert_dictionaries(entities_of_documents)
        print('>', reverted_entities)
        times, locations = Vectorizer.resolve_entities(reverted_entities)
        preprocessed_times, _, _ = Preprocesser.preprocess(times, lemmatize=False, lower=False,
                                                           pos_filter=None, remove_stopwords=False,
                                                           remove_punctuation=False,
                                                           lan_model=lan_model, ner=False)

        preprocessed_locations, _, _ = Preprocesser.preprocess(locations, lemmatize=False, lower=False,
                                                               pos_filter=None, remove_stopwords=False,
                                                               remove_punctuation=False,
                                                               lan_model=lan_model, ner=False)
        # print(preprocessed_times)

        documents = [TaggedDocument(preprocessed_document_text, [doc_id])
                     for preprocessed_document_text, doc_id in zip(preprocessed_documents, doc_ids)]
        documents.extend([TaggedDocument(preprocessed_time, [f'{doc_id}_time'])
                          for preprocessed_time, doc_id in zip(preprocessed_times, doc_ids)])
        documents.extend([TaggedDocument(preprocessed_location, [f'{doc_id}_loc'])
                          for preprocessed_location, doc_id in zip(preprocessed_locations, doc_ids)])
        print(len(documents))
        pretrained_emb = "E:/embeddings/google300.txt"
        print("Start training")
        model = Doc2Vec(documents, vector_size=300, window=10, min_count=2, workers=4, epochs=20,
                        pretrained_emb=pretrained_emb)
        # for tag in model.docvecs.doctags:
        #     if not (tag.endswith('_time') or tag.endswith('_loc')):
        #         new_vec = model.docvecs[tag] + model.docvecs[f'{tag}_time'] + model.docvecs[f'{tag}_loc']
        #         print(tag)
        #         print(model.docvecs[tag])
        #         print(new_vec)
        #     # print(model.docvecs[tag])

        path = f"{save_path}_book2vec_risch.model"
        words_dict, docs_dict = Vectorizer.model2dict(model)
        docs_dict = Vectorizer.combine_vectors(docs_dict)
        Vectorizer.my_save_doc2vec_format(fname=path, doctag_vec=docs_dict, word_vec=words_dict,
                                          prefix='*dt_',
                                          fvocab=None, binary=False)

        vecs = Vectorizer.my_load_doc2vec_format(fname=path, binary=False)

        return vecs

    @staticmethod
    def combine_vectors(document_dictionary: Dict[str, np.array]):
        summed_vecs = {}
        for tag in document_dictionary.keys():
            if not (tag.endswith('_time') or tag.endswith('_loc')):
                new_vec = document_dictionary[tag] + document_dictionary[f'{tag}_time'] + document_dictionary[f'{tag}_loc']
                summed_vecs[f'{tag}_sum'] = new_vec
            # print(model.docvecs[tag])
        summed_vecs.update(document_dictionary)
        return summed_vecs

    @staticmethod
    def resolve_entities(entities_of_documents: Union[List, Dict]):
        time_sets = []
        location_sets = []
        if isinstance(entities_of_documents, list):
            entity_iterable = entities_of_documents
        elif isinstance(entities_of_documents, dict):
            entity_iterable = entities_of_documents.values()
        else:
            raise UserWarning("Passed entities are neither in list or dict!")

        for entities_of_document in entity_iterable:
            # Time
            time_set = set(entities_of_document['DATE'])
            time_set.update(entities_of_document['TIME'])
            # TODO: special handling required
            time_set.update(entities_of_document['EVENT'])
            time_sets.append(' '.join(time_set))
            # Location
            location_set = set(entities_of_document['FAC'])
            location_set.update(entities_of_document['GPE'])
            location_set.update(entities_of_document['LOC'])
            location_sets.append(' '.join(location_set))

            # # Use Maybe
            # # Subjects and Objects
            # entities_of_document['PERSON']
            # entities_of_document['NORP']
            # entities_of_document['ORG']
            # entities_of_document['PRODUCT']
            # entities_of_document['WORK_OF_ART']
            #
            # # Unused
            # ## Numbers
            # entities_of_document['PERCENT']
            # entities_of_document['MONEY']
            # entities_of_document['QUANTITY']
            # entities_of_document['ORDINAL']
            # entities_of_document['CARDINAL']
            # ## Language
            # entities_of_document['LAW']
            # entities_of_document['LANGUAGE']

        return time_sets, location_sets

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
        docvecs = doctag_vec
        wv_vocab = word_vec  # self.wv.vocab
        wv_vectors = np.array(list(word_vec.values()))  # self.wv.vectors

        total_vec = len(wv_vocab) + len(docvecs)
        write_first_line = False
        # save word vectors
        if word_vec:
            if not doctag_vec:
                total_vec = len(wv_vocab)
            # self.wv.save_word2vec_format(fname, fvocab, binary, total_vec)
            Vectorizer.my_save_word2vec_format(fname, wv_vocab, wv_vectors, binary, total_vec)
        # save document vectors
        if doctag_vec:
            if not word_vec:
                total_vec = len(docvecs)
                write_first_line = True
            # self.docvecs.save_word2vec_format(
            #     fname, prefix=prefix, fvocab=fvocab, total_vec=total_vec,
            #     binary=binary, write_first_line=write_first_line)
            Vectorizer.my_save_docvec_word2vec_format(fname, doctag_vec, prefix, fvocab, total_vec, binary,
                                                      write_first_line)

    @staticmethod
    def my_load_doc2vec_format(fname: str, binary: bool = False):
        return DocumentKeyedVectors(KeyedVectors.load_word2vec_format(fname=fname, binary=binary))

    @staticmethod
    def show_results(model: Union[Doc2Vec, DocumentKeyedVectors], corpus: Corpus):

        # print(model.docvecs[0])
        # print(model.docvecs.doctags)
        # print(model.docvecs.distance())

        # print(model.wv.most_similar('God'))
        # print(model.wv.most_similar([model['God']]))
        # print('------')
        # print(model.wv.most_similar([model.docvecs['bs_0']]))
        # print('--')
        Vectorizer.most_similar_words_to_documents(model, positives=['bs_0'])

        # finish = 100
        # c = 0
        # for doc_id, document in corpus.documents.items():
        #     print(doc_id, corpus.id2desc(doc_id), model.wv.most_similar([model.docvecs[doc_id]]))
        #     if c > finish:
        #         break
        #     c += 1
        # print('------')
        # print(model.wv.most_similar(positive=[model.docvecs['bs_0'], model.docvecs['bs_1']],
        #                             negative=[model.docvecs['bs_2']]))
        # print('--')
        Vectorizer.most_similar_words_to_documents(model, positives=['bs_0', 'bs_1'], negatives=['bs_2'])
        # print('------')
        # for result in model.docvecs.most_similar(positive=[model.docvecs['bs_0'], model.docvecs['bs_1']],
        #                                          negative=[model.docvecs['bs_2']]):
        #     index, sim = result
        #     print(index, corpus.id2desc(index), sim)
        # print('--')
        Vectorizer.most_similar_documents_to_documents(model, corpus, positives=['bs_0', 'bs_1'], negatives=['bs_2'])

        # print('------')
        # for result in model.docvecs.most_similar([model['God']]):
        #     index, sim = result
        #     print(index, corpus.id2desc(index), sim)
        # print('--')
        Vectorizer.most_similar_documents_to_words(model, corpus, positives=['God'])

        # print('------')
        # for result in model.docvecs.most_similar(positive=[model['woman'], model['king']], negative=[model['man']]):
        #     index, sim = result
        #     print(index, corpus.id2desc(index), sim)
        # print('--')
        Vectorizer.most_similar_documents_to_words(model, corpus, positives=['woman', 'god'], negatives=['man'])
        # Vectorizer.most_similar_documents_to_words(model, corpus, positives=['queen'])

    @staticmethod
    def get_topn_of_same_type(model: Union[Doc2Vec, DocumentKeyedVectors],
                              positive_tags: List[str],
                              positive_list: List[str], negative_list: List[str],
                              topn: int,
                              feature_to_use: str = None):

        def extract_feature(tag_list):
            tag = tag_list[0]
            if not(tag.endswith('_sum') or tag.endswith('_loc') or tag.endswith('_time')):
                return 'NF'
            splitt = tag.split('_')
            print(splitt[-1])
            return f"_{splitt[-1]}"

        high_topn = topn * 10
        if feature_to_use:
            feature = feature_to_use
        else:
            feature = extract_feature(positive_tags)
        results = model.docvecs.most_similar(positive=positive_list, negative=negative_list, topn=high_topn)
        if feature == 'NF':
            results = [result for result in results if not (result[0].endswith('_sum')
                       or result[0].endswith('_loc')
                       or result[0].endswith('_time'))]
        else:
            results = [result for result in results if result[0].endswith(feature)]
        if len(results) >= topn:
            return results[:topn]
        else:
            return Vectorizer.get_topn_of_same_type(model, positive_tags, positive_list, negative_list, high_topn)[:topn]

    @staticmethod
    def most_similar_words_to_documents(model: Union[Doc2Vec, DocumentKeyedVectors],
                                        positives: List[str],
                                        negatives: List[str] = None,
                                        topn: int = 10):
        if negatives is None:
            negatives = []
        positive_list = []
        for word in positives:
            positive_list.append(model.docvecs[word])

        negative_list = []
        for word in negatives:
            negative_list.append(model.docvecs[word])
        results = model.wv.most_similar(positive=positive_list, negative=negative_list, topn=topn)
        for result in results:
            word, sim = result
            print(word, sim)

    @staticmethod
    def most_similar_documents_to_words(model: Union[Doc2Vec, DocumentKeyedVectors],
                                        corpus: Corpus, positives: List[str],
                                        negatives=None,
                                        topn: int = 10,
                                        restrict_to_same: bool = True,
                                        feature_to_use: str = None):
        if feature_to_use is None:
            feature_to_use = "_sum"
        if negatives is None:
            negatives = []
        positive_list = []
        for word in positives:
            positive_list.append(model.wv[word])

        negative_list = []
        for word in negatives:
            negative_list.append(model.wv[word])

        if restrict_to_same:
            results = Vectorizer.get_topn_of_same_type(model, positives, positive_list, negative_list, topn,
                                                       feature_to_use=feature_to_use)
        else:
            results = model.docvecs.most_similar(positive=positive_list, negative=negative_list, topn=topn)
        for result in results:
            index, sim = result
            print(index, corpus.id2desc(index), sim)

    @staticmethod
    def most_similar_documents_to_documents(model: Union[Doc2Vec, DocumentKeyedVectors],
                                            corpus: Corpus, positives: Union[List[str], str],
                                            negatives: Union[List[str], str] = None,
                                            topn: int = 10,
                                            restrict_to_same: bool = True,
                                            feature_to_use: str = None):
        positive_list = []
        if isinstance(positives, str):
            positives = [positives]

        for doc_id in positives:
            positive_list.append(model.docvecs[doc_id])

        if negatives is None:
            negatives = []
        elif isinstance(negatives, str):
            negatives = [negatives]
        else:
            pass

        negative_list = []
        for doc_id in negatives:
            negative_list.append(model.docvecs[doc_id])

        if restrict_to_same:
            results = Vectorizer.get_topn_of_same_type(model, positives, positive_list, negative_list, topn,
                                                       feature_to_use)
        else:
            results = model.docvecs.most_similar(positive=positive_list, negative=negative_list, topn=topn)
        for result in results:
            index, sim = result
            print(index, corpus.id2desc(index), sim)

    @staticmethod
    def most_similar_documents(model: Union[Doc2Vec, DocumentKeyedVectors],
                               corpus: Corpus, positives: Union[List[str], str],
                               negatives: Union[List[str], str] = None,
                               topn: int = 10,
                               restrict_to_same: bool = True,
                               feature_to_use: str = None):
        positive_list = []
        if isinstance(positives, str):
            positives = [positives]

        for doc_id in positives:
            try:
                positive_list.append(model.docvecs[doc_id])
            except KeyError:
                positive_list.append(model.wv[doc_id])

        if negatives is None:
            negatives = []
        elif isinstance(negatives, str):
            negatives = [negatives]
        else:
            pass

        negative_list = []
        for doc_id in negatives:
            try:
                negative_list.append(model.docvecs[doc_id])
            except KeyError:
                negative_list.append(model.wv[doc_id])

        if restrict_to_same:
            results = Vectorizer.get_topn_of_same_type(model, positives, positive_list, negative_list, topn,
                                                       feature_to_use)
        else:
            results = model.docvecs.most_similar(positive=positive_list, negative=negative_list, topn=topn)
        for result in results:
            index, sim = result
            print(index, corpus.id2desc(index), sim)

    @staticmethod
    def most_similar_words(model: Union[Doc2Vec, DocumentKeyedVectors],
                                        positives: List[str],
                                        negatives: List[str] = None,
                                        topn: int = 10):
        if negatives is None:
            negatives = []
        positive_list = []
        for word in positives:
            try:
                positive_list.append(model.docvecs[word])
            except KeyError:
                positive_list.append(model.wv[word])

        negative_list = []
        for word in negatives:
            try:
                negative_list.append(model.docvecs[word])
            except KeyError:
                negative_list.append(model.wv[word])
        results = model.wv.most_similar(positive=positive_list, negative=negative_list, topn=topn)
        for result in results:
            word, sim = result
            print(word, sim)


class DocumentEmbedding:
    @staticmethod
    def avg_wv2doc(corpus: Corpus, save_path: str = "models/my_model"):
        # Preprocesser.preprocess(return_in_sentence_format=True)
        # print('sents', preprocessed_sentences)
        # print(preprocessed_documents)
        _, doc_ids = corpus.get_texts_and_doc_ids()
        preprocessed_sentences = corpus.get_flat_corpus_sentences()
        preprocessed_documents = corpus.get_flat_document_tokens()

        # for d in preprocessed_documents:
        #     print(d[:10])
        # print(preprocessed_documents)
        print(f'Vectorize {len(preprocessed_sentences)} sentences and {len(preprocessed_documents)} documents')
        model = Word2Vec(preprocessed_sentences, size=100, window=5, min_count=1, workers=4, iter=20)
        docs_dict = {}
        for doc_id, doc in zip(doc_ids, preprocessed_documents):
            vector = []
            for token in doc:
                # print(token, model.wv.vocab[token])
                vector.append(model.wv[token])
            vector = sum(np.array(vector)) / len(doc)
            docs_dict[doc_id] = vector

        path = f"{save_path}_avg_wv2doc.model"
        words_dict = {word: model.wv[word] for word in model.wv.vocab}
        Vectorizer.my_save_doc2vec_format(fname=path, doctag_vec=docs_dict, word_vec=words_dict,
                                          prefix='*dt_',
                                          fvocab=None, binary=False)

        vecs = Vectorizer.my_load_doc2vec_format(fname=path, binary=False)
        return vecs

    @staticmethod
    def doc2vec(corpus: Corpus, save_path: str = "models/my_model"):
        # documents = [TaggedDocument(doc, [i])
        #              for i, doc in enumerate(Preprocesser.tokenize(corpus.get_texts_and_doc_ids()))]
        # documents = [TaggedDocument(Preprocesser.tokenize(document.text), [doc_id])
        #              for doc_id, document in corpus.documents.items()]
        _, doc_ids = corpus.get_texts_and_doc_ids()
        preprocessed_documents = corpus.get_flat_document_tokens()
        # print(preprocessed_documents)
        documents = [TaggedDocument(preprocessed_document_text, [doc_id])
                     for preprocessed_document_text, doc_id in zip(preprocessed_documents, doc_ids)]
        # print(documents[0])
        # model = Doc2Vec(documents, vector_size=100, window=10, min_count=2, workers=4, epochs=20)
        pretrained_emb = None  # "E:/embeddings"
        model = Doc2Vec(vector_size=100, min_count=2, epochs=20,
                        pretrained_emb=pretrained_emb)
        model.build_vocab(documents)

        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

        path = f"{save_path}_doc2vec.model"
        words_dict, docs_dict = Vectorizer.model2dict(model)
        Vectorizer.my_save_doc2vec_format(fname=path, doctag_vec=docs_dict, word_vec=words_dict,
                                          prefix='*dt_',
                                          fvocab=None, binary=False)

        vecs = Vectorizer.my_load_doc2vec_format(fname=path, binary=False)
        return vecs

    @staticmethod
    def book2vec_risch(corpus: Corpus, save_path: str = "models/my_model"):
        lemma = False
        lower = False
        # documents = [TaggedDocument(doc, [i])
        #              for i, doc in enumerate(Preprocesser.tokenize(corpus.get_texts_and_doc_ids()))]
        # documents = [TaggedDocument(Preprocesser.tokenize(document.text), [doc_id])
        #              for doc_id, document in corpus.documents.items()]
        lan_model = corpus.give_spacy_lan_model()
        # print('>', preprocessed_documents)
        _, doc_ids = corpus.get_texts_and_doc_ids()
        if corpus.document_entities is None:
            raise UserWarning("No Entities set!")
        reverted_entities = Utils.revert_dictionaries(corpus.document_entities)
        # print('>', reverted_entities)
        times, locations = Vectorizer.resolve_entities(reverted_entities)
        print(times)

        preprocessed_times = Preprocesser.structure_string_texts(times, lan_model, lemma=lemma, lower=lower)
        preprocessed_locations = Preprocesser.structure_string_texts(locations, lan_model, lemma=lemma, lower=lower)
        # preprocessed_times, _, _ = Preprocesser.preprocess(times, lemmatize=False, lower=False,
        #                                                    pos_filter=None, remove_stopwords=False,
        #                                                    remove_punctuation=False,
        #                                                    lan_model=lan_model, ner=False)
        #
        # preprocessed_locations, _, _ = Preprocesser.preprocess(locations, lemmatize=False, lower=False,
        #                                                        pos_filter=None, remove_stopwords=False,
        #                                                        remove_punctuation=False,
        #                                                        lan_model=lan_model, ner=False)
        print(preprocessed_times)
        # print(preprocessed_times)
        preprocessed_documents = corpus.get_flat_document_tokens(lemma=lemma, lower=lower)
        print('>', preprocessed_documents[0])
        documents = [TaggedDocument(preprocessed_document_text, [doc_id])
                     for preprocessed_document_text, doc_id in zip(preprocessed_documents, doc_ids)]
        documents.extend([TaggedDocument(preprocessed_time, [f'{doc_id}_time'])
                          for preprocessed_time, doc_id in zip(preprocessed_times, doc_ids)])
        documents.extend([TaggedDocument(preprocessed_location, [f'{doc_id}_loc'])
                          for preprocessed_location, doc_id in zip(preprocessed_locations, doc_ids)])
        print(len(documents))
        pretrained_emb = "E:/embeddings/google300.txt"
        print("Start training")
        model = Doc2Vec(documents, vector_size=300, window=10, min_count=2, workers=4, epochs=20,
                        pretrained_emb=pretrained_emb)
        # for tag in model.docvecs.doctags:
        #     if not (tag.endswith('_time') or tag.endswith('_loc')):
        #         new_vec = model.docvecs[tag] + model.docvecs[f'{tag}_time'] + model.docvecs[f'{tag}_loc']
        #         print(tag)
        #         print(model.docvecs[tag])
        #         print(new_vec)
        #     # print(model.docvecs[tag])

        path = f"{save_path}_book2vec_risch.model"
        words_dict, docs_dict = Vectorizer.model2dict(model)
        docs_dict = Vectorizer.combine_vectors(docs_dict)
        Vectorizer.my_save_doc2vec_format(fname=path, doctag_vec=docs_dict, word_vec=words_dict,
                                          prefix='*dt_',
                                          fvocab=None, binary=False)

        vecs = Vectorizer.my_load_doc2vec_format(fname=path, binary=False)

        return vecs

    @staticmethod
    def combine_vectors(document_dictionary: Dict[str, np.array]):
        summed_vecs = {}
        for tag in document_dictionary.keys():
            if not (tag.endswith('_time') or tag.endswith('_loc')):
                new_vec = document_dictionary[tag] + document_dictionary[f'{tag}_time'] + document_dictionary[
                    f'{tag}_loc']
                summed_vecs[f'{tag}_sum'] = new_vec
            # print(model.docvecs[tag])
        summed_vecs.update(document_dictionary)
        return summed_vecs

    @staticmethod
    def resolve_entities(entities_of_documents):
        time_sets = []
        location_sets = []
        for entities_of_document in entities_of_documents:
            # Time
            time_set = set(entities_of_document['DATE'])
            time_set.update(entities_of_document['TIME'])
            # TODO: special handling required
            time_set.update(entities_of_document['EVENT'])
            time_sets.append(' '.join(time_set))
            # Location
            location_set = set(entities_of_document['FAC'])
            location_set.update(entities_of_document['GPE'])
            location_set.update(entities_of_document['LOC'])
            location_sets.append(' '.join(location_set))

            # # Use Maybe
            # # Subjects and Objects
            # entities_of_document['PERSON']
            # entities_of_document['NORP']
            # entities_of_document['ORG']
            # entities_of_document['PRODUCT']
            # entities_of_document['WORK_OF_ART']
            #
            # # Unused
            # ## Numbers
            # entities_of_document['PERCENT']
            # entities_of_document['MONEY']
            # entities_of_document['QUANTITY']
            # entities_of_document['ORDINAL']
            # entities_of_document['CARDINAL']
            # ## Language
            # entities_of_document['LAW']
            # entities_of_document['LANGUAGE']

        return time_sets, location_sets

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
        docvecs = doctag_vec
        wv_vocab = word_vec  # self.wv.vocab
        wv_vectors = np.array(list(word_vec.values()))  # self.wv.vectors

        total_vec = len(wv_vocab) + len(docvecs)
        write_first_line = False
        # save word vectors
        if word_vec:
            if not doctag_vec:
                total_vec = len(wv_vocab)
            # self.wv.save_word2vec_format(fname, fvocab, binary, total_vec)
            Vectorizer.my_save_word2vec_format(fname, wv_vocab, wv_vectors, binary, total_vec)
        # save document vectors
        if doctag_vec:
            if not word_vec:
                total_vec = len(docvecs)
                write_first_line = True
            # self.docvecs.save_word2vec_format(
            #     fname, prefix=prefix, fvocab=fvocab, total_vec=total_vec,
            #     binary=binary, write_first_line=write_first_line)
            Vectorizer.my_save_docvec_word2vec_format(fname, doctag_vec, prefix, fvocab, total_vec, binary,
                                                      write_first_line)

    @staticmethod
    def my_load_doc2vec_format(fname: str, binary: bool = False):
        return DocumentKeyedVectors(KeyedVectors.load_word2vec_format(fname=fname, binary=binary))

    @staticmethod
    def show_results(model: Union[Doc2Vec, DocumentKeyedVectors], corpus: Corpus):

        # print(model.docvecs[0])
        # print(model.docvecs.doctags)
        # print(model.docvecs.distance())

        # print(model.wv.most_similar('God'))
        # print(model.wv.most_similar([model['God']]))
        # print('------')
        # print(model.wv.most_similar([model.docvecs['bs_0']]))
        # print('--')
        Vectorizer.most_similar_words_to_documents(model, positives=['bs_0'])

        # finish = 100
        # c = 0
        # for doc_id, document in corpus.documents.items():
        #     print(doc_id, corpus.id2desc(doc_id), model.wv.most_similar([model.docvecs[doc_id]]))
        #     if c > finish:
        #         break
        #     c += 1
        # print('------')
        # print(model.wv.most_similar(positive=[model.docvecs['bs_0'], model.docvecs['bs_1']],
        #                             negative=[model.docvecs['bs_2']]))
        # print('--')
        Vectorizer.most_similar_words_to_documents(model, positives=['bs_0', 'bs_1'], negatives=['bs_2'])
        # print('------')
        # for result in model.docvecs.most_similar(positive=[model.docvecs['bs_0'], model.docvecs['bs_1']],
        #                                          negative=[model.docvecs['bs_2']]):
        #     index, sim = result
        #     print(index, corpus.id2desc(index), sim)
        # print('--')
        Vectorizer.most_similar_documents_to_documents(model, corpus, positives=['bs_0', 'bs_1'], negatives=['bs_2'])

        # print('------')
        # for result in model.docvecs.most_similar([model['God']]):
        #     index, sim = result
        #     print(index, corpus.id2desc(index), sim)
        # print('--')
        Vectorizer.most_similar_documents_to_words(model, corpus, positives=['God'])

        # print('------')
        # for result in model.docvecs.most_similar(positive=[model['woman'], model['king']], negative=[model['man']]):
        #     index, sim = result
        #     print(index, corpus.id2desc(index), sim)
        # print('--')
        Vectorizer.most_similar_documents_to_words(model, corpus, positives=['woman', 'god'], negatives=['man'])
        # Vectorizer.most_similar_documents_to_words(model, corpus, positives=['queen'])

    @staticmethod
    def get_topn_of_same_type(model: Union[Doc2Vec, DocumentKeyedVectors],
                              positive_tags: List[str],
                              positive_list: List[str], negative_list: List[str],
                              topn: int,
                              feature_to_use: str = None):

        def extract_feature(tag_list):
            tag = tag_list[0]
            if not (tag.endswith('_sum') or tag.endswith('_loc') or tag.endswith('_time')):
                return 'NF'
            splitt = tag.split('_')
            print(splitt[-1])
            return f"_{splitt[-1]}"

        high_topn = topn * 10
        if feature_to_use:
            feature = feature_to_use
        else:
            feature = extract_feature(positive_tags)
        results = model.docvecs.most_similar(positive=positive_list, negative=negative_list, topn=high_topn)
        if feature == 'NF':
            results = [result for result in results if not (result[0].endswith('_sum')
                                                            or result[0].endswith('_loc')
                                                            or result[0].endswith('_time'))]
        else:
            results = [result for result in results if result[0].endswith(feature)]
        if len(results) >= topn:
            return results[:topn]
        else:
            return Vectorizer.get_topn_of_same_type(model, positive_tags, positive_list, negative_list, high_topn)[
                   :topn]

    @staticmethod
    def most_similar_words_to_documents(model: Union[Doc2Vec, DocumentKeyedVectors],
                                        positives: List[str],
                                        negatives: List[str] = None,
                                        topn: int = 10):
        if negatives is None:
            negatives = []
        positive_list = []
        for word in positives:
            positive_list.append(model.docvecs[word])

        negative_list = []
        for word in negatives:
            negative_list.append(model.docvecs[word])
        results = model.wv.most_similar(positive=positive_list, negative=negative_list, topn=topn)
        for result in results:
            word, sim = result
            print(word, sim)

    @staticmethod
    def most_similar_documents_to_words(model: Union[Doc2Vec, DocumentKeyedVectors],
                                        corpus: Corpus, positives: List[str],
                                        negatives=None,
                                        topn: int = 10,
                                        restrict_to_same: bool = True,
                                        feature_to_use: str = None):
        if feature_to_use is None:
            feature_to_use = "_sum"
        if negatives is None:
            negatives = []
        positive_list = []
        for word in positives:
            positive_list.append(model.wv[word])

        negative_list = []
        for word in negatives:
            negative_list.append(model.wv[word])

        if restrict_to_same:
            results = Vectorizer.get_topn_of_same_type(model, positives, positive_list, negative_list, topn,
                                                       feature_to_use=feature_to_use)
        else:
            results = model.docvecs.most_similar(positive=positive_list, negative=negative_list, topn=topn)
        for result in results:
            index, sim = result
            print(index, corpus.id2desc(index), sim)

    @staticmethod
    def most_similar_documents_to_documents(model: Union[Doc2Vec, DocumentKeyedVectors],
                                            corpus: Corpus, positives: Union[List[str], str],
                                            negatives: Union[List[str], str] = None,
                                            topn: int = 10,
                                            restrict_to_same: bool = True,
                                            feature_to_use: str = None):
        positive_list = []
        if isinstance(positives, str):
            positives = [positives]

        for doc_id in positives:
            positive_list.append(model.docvecs[doc_id])

        if negatives is None:
            negatives = []
        elif isinstance(negatives, str):
            negatives = [negatives]
        else:
            pass

        negative_list = []
        for doc_id in negatives:
            negative_list.append(model.docvecs[doc_id])

        if restrict_to_same:
            results = Vectorizer.get_topn_of_same_type(model, positives, positive_list, negative_list, topn,
                                                       feature_to_use)
        else:
            results = model.docvecs.most_similar(positive=positive_list, negative=negative_list, topn=topn)
        for result in results:
            index, sim = result
            print(index, corpus.id2desc(index), sim)

    @staticmethod
    def most_similar_documents(model: Union[Doc2Vec, DocumentKeyedVectors],
                               corpus: Corpus, positives: Union[List[str], str],
                               negatives: Union[List[str], str] = None,
                               topn: int = 10,
                               restrict_to_same: bool = True,
                               feature_to_use: str = None):
        positive_list = []
        if isinstance(positives, str):
            positives = [positives]

        for doc_id in positives:
            try:
                positive_list.append(model.docvecs[doc_id])
            except KeyError:
                positive_list.append(model.wv[doc_id])

        if negatives is None:
            negatives = []
        elif isinstance(negatives, str):
            negatives = [negatives]
        else:
            pass

        negative_list = []
        for doc_id in negatives:
            try:
                negative_list.append(model.docvecs[doc_id])
            except KeyError:
                negative_list.append(model.wv[doc_id])

        if restrict_to_same:
            results = Vectorizer.get_topn_of_same_type(model, positives, positive_list, negative_list, topn,
                                                       feature_to_use)
        else:
            results = model.docvecs.most_similar(positive=positive_list, negative=negative_list, topn=topn)
        for result in results:
            index, sim = result
            print(index, corpus.id2desc(index), sim)

    @staticmethod
    def most_similar_words(model: Union[Doc2Vec, DocumentKeyedVectors],
                           positives: List[str],
                           negatives: List[str] = None,
                           topn: int = 10):
        if negatives is None:
            negatives = []
        positive_list = []
        for word in positives:
            try:
                positive_list.append(model.docvecs[word])
            except KeyError:
                positive_list.append(model.wv[word])

        negative_list = []
        for word in negatives:
            try:
                negative_list.append(model.docvecs[word])
            except KeyError:
                negative_list.append(model.wv[word])
        results = model.wv.most_similar(positive=positive_list, negative=negative_list, topn=topn)
        for result in results:
            word, sim = result
            print(word, sim)