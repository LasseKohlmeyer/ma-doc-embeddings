import os
from typing import Union

from corpus_structure import DataHandler, Corpus, Preprocesser, Utils, ConfigLoader

config = ConfigLoader.get_config()


def build_series_corpus(corpus: Corpus, annotated_series_corpus_path: str, number_of_subparts: int):
    corpus = Preprocesser.filter_too_small_docs_from_corpus(corpus)
    corpus.fake_series(series_corpus_dir=annotated_series_corpus_path,
                       number_of_sub_parts=number_of_subparts)
    return Corpus.fast_load(path=annotated_series_corpus_path, load_entities=False)


def chunk_documents(data_set: str, nr_subparts: int, corpus_size: Union[int, str]):
    annotated_corpus_path = os.path.join(config["system_storage"]["corpora"],
                                         f'{data_set}')

    annotated_series_corpus_path = os.path.join(config["system_storage"]["corpora"],
                                                f'{data_set}_{nr_subparts}_'
                                                f'{corpus_size}_series')

    number_of_subparts = 2

    try:
        # check if series corpus exists
        # corpus = Corpus(annotated_series_corpus_path)
        corpus = Corpus.fast_load(path=annotated_series_corpus_path)
    except FileNotFoundError:
        try:
            # check if general corpus exists
            corpus = Corpus.fast_load(path=annotated_corpus_path, load_entities=False)
            if corpus_size != "no_limit":
                corpus = corpus.sample(corpus_size, seed=42)
            corpus = build_series_corpus(corpus,
                                         annotated_series_corpus_path,
                                         number_of_subparts)

            # corpus.save_corpus_adv(annotated_series_corpus_path)
        except FileNotFoundError:
            # load from raw data
            corpus = DataHandler.load_corpus(data_set)
            if corpus_size != "no_limit":
                corpus = corpus.sample(corpus_size, seed=42)

            Preprocesser.annotate_and_save(corpus, corpus_dir=annotated_corpus_path, without_spacy=False)
            # corpus = Preprocesser.annotate_corpus(corpus)
            # corpus.save_corpus_adv(annotated_corpus_path)

            corpus = build_series_corpus(Corpus.fast_load(path=annotated_corpus_path, load_entities=False),
                                         annotated_series_corpus_path,
                                         number_of_subparts)

    return corpus


if __name__ == '__main__':
    pass