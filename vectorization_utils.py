import os
from typing import Union, List, Dict

from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec
from gensim import utils
import numpy as np
from numpy import float32 as real
from doc2vec_structures import DocumentKeyedVectors
from corpus_structure import Corpus, ConfigLoader, DataHandler

config = ConfigLoader.get_config()


class Vectorization:
    @staticmethod
    def build_vec_file_name(number_of_subparts: Union[int, str], size: Union[int, str], dataset: str, filter_mode: str,
                            vectorization_algorithm: str, fake_series: str, allow_combination: bool = False) \
            -> str:

        combination = None
        if "_concat" in vectorization_algorithm:
            vectorization_algorithm = vectorization_algorithm.replace("_concat", "")
            combination = "con"
        elif "_pca" in vectorization_algorithm :
            vectorization_algorithm = vectorization_algorithm.replace("_pca", "")
            combination = "pca"
        elif "_tsne" in vectorization_algorithm :
            vectorization_algorithm = vectorization_algorithm.replace("_tsne", "")
            combination = "tsne"
        elif "_umap" in vectorization_algorithm :
            vectorization_algorithm = vectorization_algorithm.replace("_umap", "")
            combination = "umap"
        elif "_avg" in vectorization_algorithm :
            vectorization_algorithm = vectorization_algorithm.replace("_avg", "")
            combination = "avg"
        elif "_auto" in vectorization_algorithm:
            vectorization_algorithm = vectorization_algorithm.replace("_auto", "")
            combination = "auto"

        sub_path = DataHandler.build_config_str(number_of_subparts, size, dataset, filter_mode,
                                                vectorization_algorithm, fake_series)
        vec_path = os.path.join(config["system_storage"]["models"], f'{sub_path}.model')
        if allow_combination and combination:
            return f'{vec_path}_{combination}'
        return vec_path

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
            Vectorization.my_save_word2vec_format(fname, wv_vocab, wv_vectors, binary, total_vec)
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
            Vectorization.my_save_docvec_word2vec_format(fname, doctag_vec, prefix, fvocab, total_vec, binary,
                                                         write_first_line)

    @staticmethod
    def my_load_doc2vec_format(fname: str, binary: bool = False, combination: str = None):
        vecs = DocumentKeyedVectors(KeyedVectors.load_word2vec_format(fname=fname, binary=binary))
        if combination == "concat":
            vecs = DocumentKeyedVectors(KeyedVectors.load_word2vec_format(fname=f'{fname}_con', binary=binary))
        elif combination == "pca":
            vecs = DocumentKeyedVectors(KeyedVectors.load_word2vec_format(fname=f'{fname}_pca', binary=binary))
        elif combination == "tsne":
            vecs = DocumentKeyedVectors(KeyedVectors.load_word2vec_format(fname=f'{fname}_tsne', binary=binary))
        elif combination == "umap":
            vecs = DocumentKeyedVectors(KeyedVectors.load_word2vec_format(fname=f'{fname}_umap', binary=binary))
        elif combination == "avg":
            vecs = DocumentKeyedVectors(KeyedVectors.load_word2vec_format(fname=f'{fname}_avg', binary=binary))
        elif combination == "auto_encoder":
            vecs = DocumentKeyedVectors(KeyedVectors.load_word2vec_format(fname=f'{fname}_auto', binary=binary))
        else:
            pass
        return vecs

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
            return Vectorization.get_topn_of_same_type_recursively(model, positive_tags, positive_list, negative_list,
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

        basic_results = model.docvecs.most_similar(positive=positive_list, negative=negative_list,
                                                   topn=len(model.docvecs.doctags))

        alpha_ends = set([str(doc_id)[-1].isalpha() for doc_id, sim in basic_results])
        if not any(alpha_ends):
            feature = "NF"

        # print(feature)
        if feature == 'NF':
            results = [result for result in basic_results if result[0][-1].isdigit()]
        else:
            results = [result for result in basic_results if result[0].endswith(feature)]

            if len(results) == 0:
                print(f'Did not found {feature} in document vectors!')
                results = [result for result in basic_results if result[0][-1].isdigit()]

        results = [result for result in results if Vectorization.doctag_filter(result[0], series)]

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

        positive_list = Vectorization.get_list(positives, model, feature_to_use)
        negative_list = Vectorization.get_list(negatives, model, feature_to_use)

        if restrict_to_same:
            results = Vectorization.get_ordered_results_of_same_type(model, positives, positive_list, negative_list,
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

        positive_list = Vectorization.get_list(positives, model, feature_to_use=feature_to_use)
        negative_list = Vectorization.get_list(negatives, model, feature_to_use=feature_to_use)
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
                              binary: bool = False,
                              concat: bool = None):
        Vectorization.my_save_doc2vec_format(fname=save_path,
                                             doctag_vec=docs_dict,
                                             word_vec=words_dict,
                                             prefix='*dt_',
                                             fvocab=None,
                                             binary=binary)

        if return_vecs:
            vecs = Vectorization.my_load_doc2vec_format(fname=save_path, binary=binary, combination=concat)
            # if concat:
            #     concat_vecs = Vectorization.my_load_doc2vec_format(fname=f'{save_path}_concat', binary=binary)
            #     return vecs, concat_vecs
            return vecs
        else:
            return True
