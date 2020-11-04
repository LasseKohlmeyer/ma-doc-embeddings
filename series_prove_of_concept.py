from utils import DataHandler, Corpus, Preprocesser
from vectorization import DocumentEmbedding


class Evaluation:
    @staticmethod
    def series_eval(vectors, series_dictionary):
        pass


# Corpus +Document
corpus = DataHandler.load_litrec_books_as_corpus()
# Series:

# actual:
# series_dict = manual_dict
# corpus = corpus

# fake:
corpus, series_dict = corpus.fake_series()

# series_dict: {doc_id} -> {series_id}, series_reverse_dict: {series_id} -> [doc_id]

# common_words: {doc_id} -> [common_words]
common_words_dict = corpus.get_common_words(series_dict)
# Document-Filter: No, Common Words Del., NER Del., Nouns Del., Verbs Del., ADJ Del., Stopwords Del.
filter_fun = "common_words"
corpus = corpus.filter(mode=filter_fun)
# Embedding: Avg vec, doc2vec, simpleAspects, simpleSegments, simple A+S

corpus = Preprocesser.annotate_corpus(corpus)

vecs = DocumentEmbedding.avg_wv2doc(corpus)
# Scoring:
Evaluation.series_eval(vecs, series_dict)


# Algorithm         | Plain | Common Words Del. | NER Del.  | Nouns Del.    | Verbs Del.    | ADJ Del.  | Stopwords Del.
# Avg vec           |   x   |                   |           |               |               |           |
# + simpleAspects   |       |                   |           |               |               |           |
# + simpleSegments  |       |                   |           |               |               |           |
# + simple A + S    |       |                   |           |               |               |           |
# doc2vec           |       |                   |           |               |               |           |
# + simpleAspects   |       |                   |           |               |               |           |
# + simpleSegments  |       |                   |           |               |               |           |
# + simple A + S    |       |                   |           |               |               |           |
# x = Number of correctly identified serial book clusters compared to ...

class DocumentFilter:

    @staticmethod
    def none(input_text: str) -> str:
        return input_text

    @staticmethod
    def common_words(input_text: str, common_words: str) -> str:
        return input_text

    @staticmethod
    def ner(input_text: str) -> str:
        return input_text

    @staticmethod
    def pos(input_text: str, pos: str) -> str:
        return input_text






