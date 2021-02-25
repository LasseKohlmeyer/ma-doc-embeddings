from collections import defaultdict
from typing import List

from corpus_structure import Corpus
import numpy as np
import pandas as pd
from scipy.stats import iqr


def corpus_stats(data_sets: List[str]):
    tuples = []

    for data_set_name in data_sets:
        corpus = Corpus.fast_load("all",
                                  "no_limit",
                                  data_set_name,
                                  "no_filter",
                                  "real",
                                  load_entities=False
                                  )
        language = corpus.language
        nr_books = len(corpus.documents)

        document_tokens = [document.length for document in corpus.documents.values()]
        tokens_total = sum(document_tokens)
        tokens_avg = f'{np.mean(document_tokens):.0f} ± {np.std(document_tokens):.0f}'
        tokens_median = f'{np.median(document_tokens):.0f} ± {iqr(document_tokens):.0f}'

        document_vocab = [document.vocab_size for document in corpus.documents.values()]
        vocab_total = sum(document_vocab)
        vocab_avg = f'{np.mean(document_vocab):.0f} ± {np.std(document_vocab):.0f}'
        vocab_median = f'{np.median(document_vocab):.0f} ± {iqr(document_vocab):.0f}'

        document_sents = [document.sentences_nr for document in corpus.documents.values()]
        sents_total = sum(document_sents)
        sents_avg = f'{np.mean(document_sents):.0f} ± {np.std(document_sents):.0f}'
        sents_median = f'{np.median(document_sents):.0f} ± {iqr(document_sents):.0f}'

        author_dict = defaultdict(list)
        for doc_id, document in corpus.documents.items():
            author_dict[document.authors].append(doc_id)

        print({author: len(doc_ids) for author, doc_ids in author_dict.items() if author is not None})
        author_vals = [len(doc_ids) for author, doc_ids in author_dict.items() if author is not None]


        author_median = f'{np.median(author_vals):.0f} ± {iqr(author_vals):.0f} [{np.min(author_vals):.0f}, {np.max(author_vals):.0f}]'
        author_mean = f'{np.mean(author_vals):.2f} ± {np.std(author_vals):.2f} [{np.min(author_vals):.0f}, {np.max(author_vals):.0f}]'
        author_max = f'{np.max(author_vals):.0f}'

        print(data_set_name, "Author median iqr / mean std", author_median, author_mean)
        if corpus.series_dict and len(corpus.series_dict) > 0:
            series_vals = [len(doc_ids) for series_id, doc_ids in corpus.series_dict.items() if series_id is not None]
            series_median = f'{np.median(series_vals):.0f} ± {iqr(series_vals):.0f} [{np.min(series_vals):.0f}, {np.max(series_vals):.0f}]'
            series_mean = f'{np.mean(series_vals):.2f} ± {np.std(series_vals):.2f} [{np.min(series_vals):.0f}, {np.max(series_vals):.0f}]'
            series_max = f'{np.max(series_vals):.0f}'
            print(data_set_name, "Series median iqr / mean std", series_median, series_mean)
        else:
            series_median = "NA"
            series_mean = "NA"

        if corpus.shared_attributes_dict is None:
            corpus.calculate_documents_with_shared_attributes()
        if corpus.shared_attributes_dict["same_genres"] and len(corpus.shared_attributes_dict["same_genres"]) > 1:
            genre_vals = [len(doc_ids) for genre, doc_ids in corpus.shared_attributes_dict["same_genres"].items() if genre is not None]
            # print(genre_vals)
            genre_median = f'{np.median(genre_vals):.0f} ± {iqr(genre_vals):.0f} [{np.min(genre_vals):.0f}, {np.max(genre_vals):.0f}]'
            genre_mean = f'{np.mean(genre_vals):.2f} ± {np.std(genre_vals):.2f} [{np.min(genre_vals):.0f}, {np.max(genre_vals):.0f}]'

            print(data_set_name, "Genre median iqr / mean std", genre_median, genre_mean)
        else:
            genre_median = "NA"
            genre_mean = "NA"


        # if corpus and len(corpus.series_dict) > 0:
        #     series_median = np.median([len(doc_ids) for series_id, doc_ids in corpus.series_dict.items()])

        tuples.append((data_set_name,
                       nr_books, language,
                       tokens_total, tokens_median,
                       vocab_total, vocab_median, author_mean, series_mean, genre_mean))
    df = pd.DataFrame(tuples, columns=["Data set", "Amount of Books", "Language",
                                       "Total Tokens", "Tokens Median ± IQR",
                                       "Total Vocabulary", "Vocabulary Median ± IQR",
                                       "Books by Same Author ± STD [Min, Max]",
                                       "Books by Same Series ± STD [Min, Max]",
                                       "Books by Same Genre ± STD [Min, Max]",
                                       # "Total Sentences", "Sentences Mean [STD]", "Sentences Median [IQR]",
                                       ], index=data_sets)
    df = df.transpose()
    print(df)
    df.to_csv("results/dataset_stats/sizes.csv", index=True)
    print(df.to_latex())


if __name__ == '__main__':
    data_sets = [
        "german_books",
        "german_series",
        "goodreads_genres",
        # "dta",
        # "litrec",
        "classic_gutenberg",
        ]

    corpus_stats(data_sets)

