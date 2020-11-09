from series_prove_of_concept import Evaluation
from utils import DataHandler, Preprocesser
from vectorization import Vectorizer


# pip install -e git+git://github.com/maohbao/gensim.git@master#egg=develop
def main():
    corpus = DataHandler.load_book_summaries_as_corpus()
    # corpus = DataHandler.load_german_books_as_corpus()
    corpus_prep = Preprocesser.annotate_corpus(corpus[:100])
    corpus_prep, series_dict = corpus_prep.fake_series(number_of_sub_parts=2)
    # corpus_prep = Preprocesser.annotate_corpus(corpus_prep)
    # print(corpus_prep.get_flat_document_tokens()[0])
    corpus_prep = Preprocesser.filter(corpus_prep)
    # print(corpus_prep.get_flat_document_tokens()[0])
    # corpus_prep = Preprocesser.filter(corpus_prep, remove_stopwords=True)
    common_words = corpus_prep.get_common_words(series_dict)
    # corpus_prep = corpus_prep.filter("common_words", common_words=common_words)
    # print(corpus_prep.get_flat_document_tokens()[0])
    corpus_prep = corpus_prep.filter("ne", common_words=common_words)

    # print(corpus_prep.get_flat_document_tokens()[0])
    # print(corpus_prep.name, corpus_prep.document_entities)
    # vecs = Vectorizer.avg_wv2doc(corpus_prep)
    vecs = Vectorizer.doc2vec(corpus_prep)
    # vecs = Vectorizer.book2vec_simple(corpus_prep)
    # Evaluation.series_eval(vecs, series_dict, corpus_prep)

    # vecs = Vectorizer.avg_wv2doc(book_summaries[:100])
    # vecs = Vectorizer.doc2vec(german_books[:100])
    # vecs = Vectorizer.book2vec_simple(book_summaries[:])
    #
    # # book_summaries_model.save_word2vec_format("test.model", doctag_vec=True)
    # # book_summaries_model.save("test_save")
    #
    # words_dict, docs_dict = Vectorizer.model2dict(book_summaries_model)
    # docs_dict = Vectorizer.combine_vectors(docs_dict)
    # Vectorizer.my_save_doc2vec_format(fname="my_test.model", doctag_vec=docs_dict, word_vec=words_dict, prefix='*dt_',
    #                                   fvocab=None, binary=False)

    # --
    # vecs = Vectorizer.my_load_doc2vec_format(fname="models/my_test.model", binary=False)
    # vecs = Vectorizer.my_load_doc2vec_format(fname="models/syn_pretrained_book2vec_risch.model", binary=False)
    #
    # print('--')
    # Vectorizer.most_similar_documents_to_documents(book_summaries_model, book_summaries, ['bs_0'])
    # Vectorizer.most_similar_documents_to_documents(vecs, book_summaries, ['bs_85_sum'])
    # print('--')
    # # Vectorizer.most_similar_words_to_documents(book_summaries_model, ['bs_0'])
    # Vectorizer.most_similar_words_to_documents(vecs, ['bs_85_sum'])
    # print('--')
    # # Vectorizer.most_similar_documents_to_words(book_summaries_model, book_summaries, ['pig'])
    # Vectorizer.most_similar_documents_to_words(vecs, book_summaries, ['dragon'], feature_to_use="_sum")
    # print('--')
    # Vectorizer.most_similar_documents_to_words(vecs, book_summaries, ['ice'], feature_to_use="_sum")
    # print('--')
    # Vectorizer.most_similar_documents_to_words(vecs, book_summaries, ['dark'], feature_to_use="_sum")
    # print('--')
    # Vectorizer.most_similar_documents_to_words(vecs, book_summaries, ['potter'], feature_to_use="_sum")
    # print('--')
    # Vectorizer.most_similar_documents(vecs, book_summaries, positives=['bs_85_sum'],
    #                                   feature_to_use="_sum")
    # print('--')
    # Vectorizer.most_similar_documents(vecs, book_summaries, positives=['music'],
    #                                   feature_to_use="_sum")
    # print('--')
    # Vectorizer.most_similar_words_to_documents(vecs, ['bs_78_sum'])
    # print('--')
    # Vectorizer.most_similar_words(vecs, ['bs_78_sum', 'emperor', 'galaxy'], ['Sauron', 'bs_85_sum'])
    # print('--')
    # Vectorizer.most_similar_documents(vecs, book_summaries,
    #                                   positives=['bs_78_sum', 'Emperor'],
    #                                   negatives= ['Sauron'],
    #                                   feature_to_use="_sum")

    # Vectorizer.most_similar_documents(vecs, corpus_prep,
    #                                   positives=['78_0'],
    #                                   feature_to_use="_sum")

    # --

    # --
    # vecs = Vectorizer.my_load_doc2vec_format(fname="models/my_model_book2vec_risch.model", binary=False)
    # vecs = Vectorizer.my_load_doc2vec_format(fname="models/my_model_doc2vec.model", binary=False)

    # print('--')
    # # Vectorizer.most_similar_documents_to_documents(book_summaries_model, book_summaries, ['bs_0'])
    # Vectorizer.most_similar_documents_to_documents(vecs, german_books, ['gfo_85'])
    # print('--')
    # # Vectorizer.most_similar_words_to_documents(book_summaries_model, ['bs_0'])
    # Vectorizer.most_similar_words_to_documents(vecs, ['gfo_85'])
    # print('--')
    # # Vectorizer.most_similar_documents_to_words(book_summaries_model, book_summaries, ['pig'])
    # Vectorizer.most_similar_documents_to_words(vecs, german_books, ['Laboratorium'], feature_to_use="NF")
    # print('--')
    # --


    # Doc2VecKeyedVectors.most_similar()

    # print(dv)
    # print(book_summaries_model.docvecs.doctags)
    # Vectorizer.show_results(book_summaries_model, book_summaries)
    # texts, entities = Preprocesser.preprocess(["das ist ein toller Test von Heinrich dem Löwen aus dem Jahr 1492. ",
    #                                            "Das ist ein anderes Textdokument von Steve Jobs aus New York!"],
    #                                           lan_model=spacy.load("de_core_news_sm"))
    # print(texts)
    # print(entities)


if __name__ == '__main__':
    main()

    # chunked_texts, chunk_list = Preprocesser.chunk_text(["hallo das ist ein mittellanger Text", "das ist kurz",
    #                                                      "das ist ein sehr viel längerer text der wirklich nichts von dem enthält, was er verspricht. wirklich gar nichts, überhaupt nichts"],
    #                                                     2)
    # print(chunked_texts)
    # print(chunk_list)
    # Preprocesser.merge_chunks(chunked_texts, chunk_list)
