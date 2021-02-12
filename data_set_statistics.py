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
        tokens_avg = f'{np.mean(document_tokens):.0f} {np.std(document_tokens):.0f}'
        tokens_median = f'{np.median(document_tokens):.0f} ± {iqr(document_tokens):.0f}'

        document_vocab = [document.vocab_size for document in corpus.documents.values()]
        vocab_total = sum(document_vocab)
        vocab_avg = f'{np.mean(document_vocab):.0f} {np.std(document_vocab):.0f}'
        vocab_median = f'{np.median(document_vocab):.0f} ± {iqr(document_vocab):.0f}'

        document_sents = [document.sentences_nr for document in corpus.documents.values()]
        sents_total = sum(document_sents)
        sents_avg = f'{np.mean(document_sents):.0f} {np.std(document_sents):.0f}'
        sents_median = f'{np.median(document_sents):.0f} ± {iqr(document_sents):.0f}'

        tuples.append((data_set_name,
                       nr_books, language,
                       tokens_total, tokens_median,
                       vocab_total, vocab_median))
    df = pd.DataFrame(tuples, columns=["Data set", "Amount of Books", "Language",
                                       "Total Tokens", "Tokens Median ± IQR",
                                       "Total Vocabulary", "Vocabulary Median ± IQR",
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
        # "dta",
        # "litrec",
        "classic_gutenberg",
        ]

    corpus_stats(data_sets)

