import json
from collections import defaultdict
from typing import List, Dict, Union

import gensim

from gensim.corpora import Dictionary
from gensim.models.doc2vec import TaggedDocument
from corpus_structure import Corpus, Document
from text_summarisation import Summarizer


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
                                 summary_dict: Union[None, Dict[str, List[Union[str, int]]]]):
    document.load_sentences_from_disk()
    if document.doc_entities is None:
        document.set_entities()

    if document.doc_entities is None:
        raise UserWarning("No Entities set!")

    times, locations = resolve_doc_entities(document.get_document_entities_representation(lemma,
                                                                                          lower))
    # print(len(times), times)

    doc_aspects = {}
    if "time" not in disable_aspects:
        doc_aspects['time'] = times

    if "loc" not in disable_aspects:
        doc_aspects['loc'] = locations

    if "raw" not in disable_aspects:
        doc_aspects['raw'] = document.get_flat_document_tokens(lemma=lemma, lower=lower)

    if "atm" not in disable_aspects:
        doc_aspects['atm'] = document.get_flat_and_filtered_document_tokens(lemma=lemma,
                                                                            lower=lower,
                                                                            pos=["ADJ", "ADV"])
    if "sty" not in disable_aspects:
        doc_aspects['sty'] = document.get_flat_and_filtered_document_tokens(lemma=lemma,
                                                                            lower=lower,
                                                                            focus_stopwords=True)
    if "cont" not in disable_aspects:
        doc_aspects["cont"] = topic_dict[doc_id]

    if "plot" not in disable_aspects:
        doc_aspects["plot"] = Summarizer.document_summary_list(document,
                                                               summary_dict,
                                                               lemma=lemma,
                                                               lower=lower)
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


class CorpusDocumentIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower

    def __len__(self):
        return len(self.corpus.documents)

    def __iter__(self):
        for doc_id, document in self.corpus.documents.items():
            yield document.get_flat_tokens_from_disk(lemma=self.lemma, lower=self.lower)


class CorpusTaggedDocumentIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower

    def __len__(self):
        return len(self.corpus.documents)

    def __iter__(self):
        for doc_id, document in self.corpus.documents.items():
            yield TaggedDocument(document.get_flat_tokens_from_disk(lemma=self.lemma, lower=self.lower), [doc_id])


class CorpusDocumentSentenceIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower

    def __len__(self):
        return len(self.corpus.documents)

    def __iter__(self):
        for doc_id, document in self.corpus.documents.items():
            # yield document.get_flat_tokens_from_disk(lemma=self.lemma, lower=self.lower)
            yield [sentence.representation(self.lemma, self.lower)
                   for sentence in document.get_sentences_from_disk()]


class FlairDocumentIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower

    def __len__(self):
        return len(self.corpus.documents)

    def __iter__(self):
        for doc_id, document in self.corpus.documents.items():
            yield doc_id, ' '.join(document.get_flat_tokens_from_disk(lemma=self.lemma, lower=self.lower))


class CorpusTaggedFacetIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False, disable_aspects: List[str] = None,
                 topic_dict: Dict = None, summary_dict: Dict = None):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower
        self.disable_aspects = disable_aspects
        self.doc_aspects = {}
        self.topic_dict = topic_dict
        self.summary_dict = summary_dict

    def __len__(self):
        return len(self.corpus.documents)

    def __iter__(self):
        for doc_id, document in self.corpus.documents.items():

            doc_aspects = calculate_facets_of_document(document,
                                                       doc_id=doc_id,
                                                       disable_aspects=self.disable_aspects,
                                                       lemma=self.lemma,
                                                       lower=self.lower,
                                                       topic_dict=self.topic_dict,
                                                       summary_dict=self.summary_dict)
            self.doc_aspects[doc_id] = {aspect_name: len(document_aspects)
                                        for aspect_name, document_aspects in doc_aspects.items()}
            for aspect_name, document_aspects in doc_aspects.items():
                yield TaggedDocument(document_aspects, [f'{doc_id}_{aspect_name}'])


class FlairFacetIterator(object):
    def __init__(self, corpus: Corpus, lemma: bool = False, lower: bool = False, disable_aspects: List[str] = None,
                 topic_dict: Dict = None, summary_dict: Dict = None):
        self.corpus = corpus
        self.lemma = lemma
        self.lower = lower
        self.disable_aspects = disable_aspects
        self.doc_aspects = {}
        self.topic_dict = topic_dict
        self.summary_dict = summary_dict

    def __len__(self):
        return len(self.corpus.documents)

    def __iter__(self):
        for doc_id, document in self.corpus.documents.items():

            doc_aspects = calculate_facets_of_document(document,
                                                       doc_id=doc_id,
                                                       disable_aspects=self.disable_aspects,
                                                       lemma=self.lemma,
                                                       lower=self.lower,
                                                       topic_dict=self.topic_dict,
                                                       summary_dict=self.summary_dict)
            self.doc_aspects[doc_id] = {aspect_name: len(document_aspects)
                                        for aspect_name, document_aspects in doc_aspects.items()}
            for aspect_name, document_aspects in doc_aspects.items():
                doc_aspect_id = f'{doc_id}_{aspect_name}'
                print(doc_aspect_id, len(document_aspects))
                yield doc_aspect_id, f"{' '.join(document_aspects)} ."

#  - facette analyzes: done
#  - content facet: done
#  - summarry facet: done
#  - heideltime: done
#  - simplify german books vs german series
#  - small test corpus with series + no series elements
#  - design common facet book task