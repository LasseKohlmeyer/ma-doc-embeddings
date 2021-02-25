import json
from collections import defaultdict
from typing import List, Dict, Union

import gensim

from gensim.corpora import Dictionary
from gensim.models.doc2vec import TaggedDocument
from corpus_structure import Corpus, Document
from text_summarisation import Summarizer
from wordnet_utils import NetWords


def resolve_entities(entities_of_documents):
    time_sets = []
    location_sets = []
    for doc_id, entities_of_document_dict in entities_of_documents.items():
        # Time
        # entities_of_document = entities_of_document_dict
        entities_of_document = defaultdict(str, entities_of_document_dict)
        # ToDo: set or list?
        time_set = list(entities_of_document['DATE'])
        time_set.extend(entities_of_document['TIME'])
        # TODO: special handling required
        time_set.extend(entities_of_document['EVENT'])
        time_sets.append(' '.join(time_set))
        # Location
        location_set = list(entities_of_document['FAC'])
        location_set.extend(entities_of_document['GPE'])
        location_set.extend(entities_of_document['LOC'])
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


def resolve_doc_entities(entities_of_document):
    # Time
    # ToDo: set or list?
    time_set = list(entities_of_document['DATE'])
    time_set.extend(entities_of_document['TIME'])
    # TODO: special handling required
    time_set.extend(entities_of_document['EVENT'])
    # Location
    location_set = list(entities_of_document['FAC'])
    location_set.extend(entities_of_document['GPE'])
    location_set.extend(entities_of_document['LOC'])

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

    return time_set, location_set


def write_aspect_frequency_analyzis(aspects: Dict[str, List[List[str]]], doc_ids: List[str], save_name: str):
    document_aspect_dict = defaultdict(dict)
    for aspect_name, aspect_documents in aspects.items():
        for doc_id, document in zip(doc_ids, aspect_documents):
            document_aspect_dict[doc_id].update({aspect_name: len(document)})

    with open(f'aspects_old/{save_name}.json', 'w', encoding="utf-8") as fp:
        json.dump(document_aspect_dict, fp, indent=1)
    return document_aspect_dict


def write_doc_based_aspect_frequency_analyzis(document_aspect_dict: Dict[str, Dict[str, List]], save_name: str):
    with open(f'aspects/{save_name}.json', 'w', encoding="utf-8") as fp:
        json.dump(document_aspect_dict, fp, indent=1)
    return document_aspect_dict


def calculate_facets_of_document(document: Document,
                                 doc_id: str,
                                 disable_aspects: List[str],
                                 lemma: bool, lower: bool,
                                 topic_dict: Union[None, Dict[str, List[str]]],
                                 summary_dict: Union[None, Dict[str, List[Union[str, int]]]],
                                 window: int = 0):
    def windowing(facet_ids, doc: Document, window_size: int):
        facet_words = []
        for (sentence_id, token_id) in facet_ids:
            # print(sentence_id, token_id, facet_ids)
            sentence = doc.sentences[sentence_id]
            lower_bound = token_id - window_size
            upper_bound = token_id + window_size
            if lower_bound < 0:
                lower_bound = 0
            if upper_bound >= len(sentence.tokens):
                upper_bound = len(sentence.tokens) - 1
            # print(lower_bound, upper_bound)
            for i in range(lower_bound, upper_bound + 1):
                # print(window, i, sentence.tokens[i].representation(lemma=lemma, lower=lower))
                facet_words.append(sentence.tokens[i].representation(lemma=lemma, lower=lower))
        # print(window_size, facet_words)
        return facet_words
    # print('w', window)
    document.load_sentences_from_disk()
    if document.doc_entities is None:
        document.set_entities()

    if document.doc_entities is None:
        raise UserWarning("No Entities set!")

    times, locations = resolve_doc_entities(document.get_document_entities_representation(lemma,
                                                                                          lower,
                                                                                          as_id=True))

    # print("times", times)
    # print("locations", locations)

    # print(len(times), times)
    doc_aspects = {}
    use_dictionary_lookup = False
    if "time" not in disable_aspects:
        times = set(times)
        if use_dictionary_lookup:
            times.update(document.get_wordnet_matches(NetWords.get_time_words(lan=document.language),
                                                      as_id=True,
                                                      lemma=lemma,
                                                      lower=lower))
        doc_aspects['time'] = windowing(facet_ids=times, doc=document, window_size=window)

    if "loc" not in disable_aspects:
        locations = set(locations)
        if use_dictionary_lookup:
            locations.update(document.get_wordnet_matches(NetWords.get_location_words(lan=document.language),
                                                          as_id=True,
                                                          lemma=lemma,
                                                          lower=lower))
        doc_aspects['loc'] = windowing(facet_ids=locations, doc=document, window_size=window)

    if "raw" not in disable_aspects:
        doc_aspects['raw'] = document.get_flat_document_tokens(lemma=lemma, lower=lower)

    if "atm" not in disable_aspects:
        atmosphere_words = document.get_flat_and_filtered_document_tokens(lemma=lemma,
                                                                          lower=lower,
                                                                          pos=["ADJ", "ADV"],
                                                                          ids=True)
        # print("atmosphere", atmosphere_words)
        atmosphere_words = set(atmosphere_words)
        if use_dictionary_lookup:
            atmosphere_words.update(document.get_wordnet_matches(NetWords.get_atmosphere_words(lan=document.language),
                                                                 as_id=True,
                                                                 lemma=lemma,
                                                                 lower=lower))

        doc_aspects['atm'] = windowing(facet_ids=atmosphere_words,
                                       doc=document,
                                       window_size=window)
    if "sty" not in disable_aspects:
        doc_aspects['sty'] = windowing(facet_ids=document.get_flat_and_filtered_document_tokens(lemma=lemma,
                                                                                                lower=lower,
                                                                                                focus_stopwords=True,
                                                                                                ids=True),
                                       doc=document,
                                       window_size=window)
    if "cont" not in disable_aspects:
        doc_aspects["cont"] = topic_dict[doc_id]

    if "plot" not in disable_aspects:
        doc_aspects["plot"] = Summarizer.document_summary_list(document,
                                                               summary_dict,
                                                               lemma=lemma,
                                                               lower=lower)
    else:
        plot_words = document.get_flat_and_filtered_document_tokens(lemma=lemma,
                                                                    lower=lower,
                                                                    pos=["VERB", "ADV"],
                                                                    ids=True)
        doc_aspects['plot'] = windowing(facet_ids=plot_words,
                                        doc=document,
                                        window_size=window)
    # print(doc_aspects.keys(), disable_aspects)
    # for key, values in doc_aspects.items():
    #     for doc_list, doc_id in zip(values, doc_ids):
    #         print(key, doc_id, doc_list[:10])

    assert set(doc_aspects.keys()).union(disable_aspects) == {'time', 'loc',
                                                              'raw',
                                                              'atm', 'sty',
                                                              'cont', 'plot'
                                                              }
    document.sentences = None
    document.doc_entities = None

    return doc_aspects


class TokenIterator(object):
    def __init__(self, corpus: Corpus,
                 lemma: bool = False, lower: bool = False):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower

    def __len__(self):
        return len(self.corpus.get_corpus_vocab())

    def __iter__(self):
        for doc_id, document in self.corpus.documents.items():
            for token in document.get_vocab(from_disk=True, lemma=self.lemma, lower=self.lower, lda_mode=True):
                if token != 'del':
                    yield token


class TopicModellingIterator(object):
    def __init__(self, corpus: Corpus, id2word_dict: Dictionary,
                 lemma: bool = False, lower: bool = False,
                 bigram_min: int = 5,
                 bigram_max: int = 100,
                 trigram_max: int = 150):
        self.corpus = corpus
        self.id2word_dict = id2word_dict
        self.lemma = lemma
        self.lower = lower
        self.bigram_min = bigram_min
        self.bigram_max = bigram_max
        self.trigram_max = trigram_max
        self.doc_ids = []

    def __len__(self):
        return len(self.corpus.documents)

    def __iter__(self):
        for doc_id, document in self.corpus.documents.items():
            document_tokens = document.get_flat_and_lda_filtered_tokens(lemma=self.lemma, lower=self.lower)

            # higher max fewer phrases.
            bigram = gensim.models.Phrases(document_tokens, min_count=self.bigram_min, threshold=self.bigram_max)

            bigram_mod = gensim.models.phrases.Phraser(bigram)

            trigram = gensim.models.Phrases(bigram[document_tokens], threshold=self.trigram_max)
            trigram_mod = gensim.models.phrases.Phraser(trigram)

            data_lemmatized = trigram_mod[bigram_mod[document_tokens]]

            # print('>', data_lemmatized)
            if doc_id not in self.doc_ids:
                self.doc_ids.append(doc_id)
            yield self.id2word_dict.doc2bow(data_lemmatized)


class CorpusSentenceIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower

    def __len__(self):
        return len(self.corpus.documents)

    def __iter__(self):
        # for fname in os.listdir(self.dirname):
        #     for line in open(os.path.join(self.dirname, fname)):
        #         yield line.split()

        for doc_id, document in self.corpus.documents.items():
            for sentence in document.get_sentences_from_disk():
                yield sentence.representation(self.lemma, self.lower)


class CorpusTaggedSentenceIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False, sentence_nr: int = None):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower
        self.sentence_nr = sentence_nr

    def __len__(self):
        return len(self.corpus.documents)

    def __iter__(self):
        # for fname in os.listdir(self.dirname):
        #     for line in open(os.path.join(self.dirname, fname)):
        #         yield line.split()

        for doc_id, document in self.corpus.documents.items():
            for sent_id, sentence in enumerate(document.get_sentences_from_disk()[:self.sentence_nr]):
                yield TaggedDocument(sentence.representation(self.lemma, self.lower), [f'{doc_id}_{sent_id}'])


class CorpusDocumentIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower
        self.doc_ids = []

    def __len__(self):
        return len(self.corpus.documents)

    def __iter__(self):
        for doc_id, document in self.corpus.documents.items():
            yield document.get_flat_tokens_from_disk(lemma=self.lemma, lower=self.lower)
            if doc_id not in self.doc_ids:
                self.doc_ids.append(doc_id)


class CorpusPlainDocumentIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower
        self.doc_ids = []

    def __len__(self):
        return len(self.corpus.documents)

    def __iter__(self):
        for doc_id, document in self.corpus.documents.items():
            yield ' '.join(document.get_flat_tokens_from_disk(lemma=self.lemma, lower=self.lower))
            if doc_id not in self.doc_ids:
                self.doc_ids.append(doc_id)


class CorpusTaggedDocumentIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False, chunk_len: int = None):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower
        self.chunk_len = chunk_len

    def __len__(self):
        return len(self.corpus.documents)

    def __iter__(self):
        if self.chunk_len:
            for doc_id, document in self.corpus.documents.items():
                tokens = document.get_flat_tokens_from_disk(lemma=self.lemma, lower=self.lower)
                for i in range(0, len(tokens), self.chunk_len):
                    chunked_tokens = tokens[i:i + self.chunk_len]
                    # print(doc_id, i, len(chunked_tokens), chunked_tokens[:10])
                    yield TaggedDocument(chunked_tokens, [f'{doc_id}_{i}'])
        else:
            for doc_id, document in self.corpus.documents.items():
                yield TaggedDocument(document.get_flat_tokens_from_disk(lemma=self.lemma, lower=self.lower), [doc_id])


# class CorpusChunkLongDocumentIterator(object):
#     def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False):
#         self.corpus = corpus
#         self.lemma = lemma
#         self.lower = lower
#         self.chunk_len = 10000
#
#     def __len__(self):
#         return len(self.corpus.documents)
#
#     def __iter__(self):
#         for doc_id, document in self.corpus.documents.items():
#             tokens = document.get_flat_tokens_from_disk(lemma=self.lemma, lower=self.lower)
#             for i in range(0, len(tokens), self.chunk_len):
#                 chunked_tokens = tokens[i:i + self.chunk_len]
#                 # print(doc_id, i, len(chunked_tokens), chunked_tokens[:10])
#                 yield TaggedDocument(chunked_tokens, [f'{doc_id}_{i}'])
#             # yield TaggedDocument(tokens, [doc_id])


# class CorpusDocumentSentenceIterator(object):
#     def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False):
#         self.corpus = corpus
#         self.lemma = lemma
#         self.lower = lower
#
#     def __len__(self):
#         return len(self.corpus.documents)
#
#     def __iter__(self):
#         for doc_id, document in self.corpus.documents.items():
#             # yield document.get_flat_tokens_from_disk(lemma=self.lemma, lower=self.lower)
#             yield [sentence.representation(self.lemma, self.lower)
#                    for sentence in document.get_sentences_from_disk()]


class FlairDocumentIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False, chunk_len: int = None):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower
        self.chunk_len = chunk_len

    def __len__(self):
        return len(self.corpus.documents)

    def __iter__(self):
        if self.chunk_len:
            for doc_id, document in self.corpus.documents.items():
                tokens = document.get_flat_tokens_from_disk(lemma=self.lemma, lower=self.lower)
                for i in range(0, len(tokens), self.chunk_len):
                    chunked_tokens = tokens[i:i + self.chunk_len]
                    # print(doc_id, i, len(chunked_tokens), chunked_tokens[:10])
                    yield f'{doc_id}_{i}', ' '.join(chunked_tokens)
        else:
            for doc_id, document in self.corpus.documents.items():
                yield doc_id, ' '.join(document.get_flat_tokens_from_disk(lemma=self.lemma, lower=self.lower))


class FlairSentenceDocumentIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False, sentence_nr: int = None):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower
        self.sentence_nr = sentence_nr

    def __len__(self):
        if self.sentence_nr:
            return sum([self.sentence_nr for _ in self.corpus.documents.values()])
        else:
            return sum([document.sentences_nr for document in self.corpus.documents.values()])

        # return len(self.corpus.documents)

    def __iter__(self):
        for doc_id, document in self.corpus.documents.items():
            for i, sentence in enumerate(document.get_sentences_from_disk()[:self.sentence_nr]):
                yield f'{doc_id}_{i}', ' '.join(sentence.representation(lemma=self.lemma, lower=self.lower))


# class FlairChunkLongDocumentIterator(object):
#     def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False, chunk_len: int = 512):
#         self.corpus = corpus
#         self.lemma = lemma
#         self.lower = lower
#         self.chunk_len = chunk_len
#
#     def __len__(self):
#         return len(self.corpus.documents)
#
#     def __iter__(self):
#         for doc_id, document in self.corpus.documents.items():
#             tokens = document.get_flat_tokens_from_disk(lemma=self.lemma, lower=self.lower)
#             for i in range(0, len(tokens), self.chunk_len):
#                 chunked_tokens = tokens[i:i + self.chunk_len]
#                 # print(doc_id, i, len(chunked_tokens), chunked_tokens[:10])
#                 yield f'{doc_id}_{i}', ' '.join(chunked_tokens)


class CorpusTaggedFacetIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False, disable_aspects: List[str] = None,
                 topic_dict: Dict = None, summary_dict: Dict = None, chunk_len: int = None,
                 facets_of_chunks: bool = True, window: int = 0):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower
        self.disable_aspects = disable_aspects
        self.doc_aspects = {}
        self.detailed_aspects = {}
        self.topic_dict = topic_dict
        self.summary_dict = summary_dict
        self.chunk_len = chunk_len
        self.facets_of_chunks = facets_of_chunks
        self.window = window

    def __len__(self):
        return len(self.corpus.documents)

    def build_doc_aspects(self, doc_id, doc_aspects):
        self.doc_aspects[doc_id] = {aspect_name: len(document_aspects)
                                    for aspect_name, document_aspects in doc_aspects.items()}
        self.detailed_aspects[doc_id] = {aspect_name: (len(document_aspects), len(set(document_aspects)),
                                                       self.corpus.documents[doc_id].length,
                                                       self.corpus.documents[doc_id].vocab_size)
                                         for aspect_name, document_aspects in doc_aspects.items()}

    def __iter__(self):
        if self.chunk_len:
            if self.facets_of_chunks:
                for doc_id, document in self.corpus.documents.items():
                    document_chunks = document.into_chunks(chunk_size=self.chunk_len)
                    for i, document_chunk in enumerate(document_chunks):
                        doc_aspects = calculate_facets_of_document(document_chunk,
                                                                   doc_id=doc_id,
                                                                   disable_aspects=self.disable_aspects,
                                                                   lemma=self.lemma,
                                                                   lower=self.lower,
                                                                   topic_dict=self.topic_dict,
                                                                   summary_dict=self.summary_dict,
                                                                   window=self.window)
                        chunked_doc_aspects = doc_aspects
                        self.build_doc_aspects(document_chunk.doc_id, doc_aspects)

                        for aspect_name, document_aspects in chunked_doc_aspects.items():
                            # print(doc_id, i, len(chunked_aspect), chunked_aspect[:10])
                            chunked_aspect_doc_id = f'{doc_id}_{aspect_name}_{i}'
                            yield TaggedDocument(document_aspects, [chunked_aspect_doc_id])

            else:
                for doc_id, document in self.corpus.documents.items():

                    doc_aspects = calculate_facets_of_document(document,
                                                               doc_id=doc_id,
                                                               disable_aspects=self.disable_aspects,
                                                               lemma=self.lemma,
                                                               lower=self.lower,
                                                               topic_dict=self.topic_dict,
                                                               summary_dict=self.summary_dict,
                                                               window=self.window)
                    self.build_doc_aspects(doc_id, doc_aspects)
                    # self.doc_aspects[doc_id] = {aspect_name: len(document_aspects)
                    #                             for aspect_name, document_aspects in doc_aspects.items()}
                    for aspect_name, document_aspects in doc_aspects.items():
                        for i in range(0, len(document_aspects), self.chunk_len):
                            chunked_aspect = document_aspects[i:i + self.chunk_len]
                            # print(doc_id, i, len(chunked_aspect), chunked_aspect[:10])
                            yield TaggedDocument(chunked_aspect, [f'{doc_id}_{aspect_name}_{i}'])
        else:
            for doc_id, document in self.corpus.documents.items():

                doc_aspects = calculate_facets_of_document(document,
                                                           doc_id=doc_id,
                                                           disable_aspects=self.disable_aspects,
                                                           lemma=self.lemma,
                                                           lower=self.lower,
                                                           topic_dict=self.topic_dict,
                                                           summary_dict=self.summary_dict,
                                                           window=self.window)
                # self.doc_aspects[doc_id] = {aspect_name: len(document_aspects)
                #                             for aspect_name, document_aspects in doc_aspects.items()}
                self.build_doc_aspects(doc_id, doc_aspects)

                for aspect_name, document_aspects in doc_aspects.items():
                    yield TaggedDocument(document_aspects, [f'{doc_id}_{aspect_name}'])


class FlairFacetIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False, disable_aspects: List[str] = None,
                 topic_dict: Dict = None, summary_dict: Dict = None, chunk_len: int = None,
                 facets_of_chunks: bool = True, window: int = 0):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower
        self.disable_aspects = disable_aspects
        self.doc_aspects = {}
        self.detailed_aspects = {}
        self.topic_dict = topic_dict
        self.summary_dict = summary_dict
        self.chunk_len = chunk_len
        self.facets_of_chunks = facets_of_chunks
        self.window = window

    def __len__(self):
        return len(self.corpus.documents)

    def build_doc_aspects(self, doc_id, doc_aspects):
        self.doc_aspects[doc_id] = {aspect_name: len(document_aspects)
                                    for aspect_name, document_aspects in doc_aspects.items()}
        self.detailed_aspects[doc_id] = {aspect_name: (len(document_aspects), len(set(document_aspects)),
                                                       self.corpus.documents[doc_id].length,
                                                       self.corpus.documents[doc_id].vocab_size)
                                         for aspect_name, document_aspects in doc_aspects.items()}
        return self.doc_aspects[doc_id]

    def __iter__(self):
        if self.chunk_len:
            if self.facets_of_chunks:
                for doc_id, document in self.corpus.documents.items():
                    document_chunks = document.into_chunks(chunk_size=self.chunk_len)
                    for i, document_chunk in enumerate(document_chunks):
                        doc_aspects = calculate_facets_of_document(document_chunk,
                                                                   doc_id=doc_id,
                                                                   disable_aspects=self.disable_aspects,
                                                                   lemma=self.lemma,
                                                                   lower=self.lower,
                                                                   topic_dict=self.topic_dict,
                                                                   summary_dict=self.summary_dict,
                                                                   window=self.window)
                        # chunked_doc_aspects = {aspect_name: len(document_aspects)
                        #                        for aspect_name, document_aspects in doc_aspects.items()}
                        # self.doc_aspects[document_chunk.doc_id] = chunked_doc_aspects

                        chunked_doc_aspects = self.build_doc_aspects(document_chunk.doc_id, doc_aspects)
                        for aspect_name, document_aspects in chunked_doc_aspects.items():
                            # print(doc_id, i, len(chunked_aspect), chunked_aspect[:10])
                            chunked_aspect_doc_id = f'{doc_id}_{aspect_name}_{i}'
                            yield chunked_aspect_doc_id, f"{' '.join(chunked_doc_aspects)} ."

            else:
                for doc_id, document in self.corpus.documents.items():

                    doc_aspects = calculate_facets_of_document(document,
                                                               doc_id=doc_id,
                                                               disable_aspects=self.disable_aspects,
                                                               lemma=self.lemma,
                                                               lower=self.lower,
                                                               topic_dict=self.topic_dict,
                                                               summary_dict=self.summary_dict,
                                                               window=self.window)
                    # self.doc_aspects[doc_id] = {aspect_name: len(document_aspects)
                    #                             for aspect_name, document_aspects in doc_aspects.items()}
                    self.build_doc_aspects(doc_id, doc_aspects)
                    for aspect_name, document_aspects in doc_aspects.items():
                        for i in range(0, len(document_aspects), self.chunk_len):
                            chunked_aspect = document_aspects[i:i + self.chunk_len]
                            # print(doc_id, i, len(chunked_aspect), chunked_aspect[:10])
                            doc_aspect_id = f'{doc_id}_{aspect_name}_{i}'
                            yield doc_aspect_id, f"{' '.join(chunked_aspect)} ."
        else:
            for doc_id, document in self.corpus.documents.items():

                doc_aspects = calculate_facets_of_document(document,
                                                           doc_id=doc_id,
                                                           disable_aspects=self.disable_aspects,
                                                           lemma=self.lemma,
                                                           lower=self.lower,
                                                           topic_dict=self.topic_dict,
                                                           summary_dict=self.summary_dict,
                                                           window=self.window)
                # self.doc_aspects[doc_id] = {aspect_name: len(document_aspects)
                #                             for aspect_name, document_aspects in doc_aspects.items()}
                self.build_doc_aspects(doc_id, doc_aspects)
                for aspect_name, document_aspects in doc_aspects.items():
                    doc_aspect_id = f'{doc_id}_{aspect_name}'
                    print(doc_aspect_id, len(document_aspects))
                    yield doc_aspect_id, f"{' '.join(document_aspects)} ."

#  - facette analyzes: done
#  - content facet: done
#  - summarry facet: done
#  - heideltime: done
#  - design common facet book task: done
#  - design facet evaluation tasks: mostly done
#  - classic corpus parser, annotation, experiments: done
#  - simplify german books vs german series
#  - small test corpus with series + no series elements
#  - doc2vec u.ä. verfahrens limits berücksichtigen über avg
#
