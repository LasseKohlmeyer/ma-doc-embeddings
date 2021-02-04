import os

from aux_utils import ConfigLoader
from corpus_structure import Corpus, DataHandler, Preprocesser
import re

from document_segments import chunk_documents


def series_dict_from_series_corpus():
    corpus = Corpus.fast_load(path="corpora/german_series", load_entities=False)
    big_corpus = Corpus.fast_load(path="corpora/german_books", load_entities=False)
    nr_matches = 0
    nr_series_books = 0
    mapped_doc_ids = {}
    for series_id, book_ids in corpus.series_dict.items():
        for book_id in book_ids:
            nr_series_books += 1
            current_author = corpus.documents[book_id].authors
            current_title = corpus.documents[book_id].title
            print(series_id, book_id, current_author, current_title)
            relevant_id = None
            for doc_id, document in big_corpus.documents.items():
                if document.authors == current_author:
                    title = document.title
                    # print(title)
                    title = f'{title} '
                    title = title.replace("Erster Band", " Band 1").replace("Zweiter Band", " Band 2") \
                        .replace("Dritter Band", " Band 3").replace("Vierter Band", " Band 4") \
                        .replace("Fünfter Band", " Band 5").replace("Sechster Band", " Band 6") \
                        .replace("Siebter Band", " Band 7").replace("Achter Band", " Band 8") \
                        .replace("Neunter Band", " Band 9").replace("Zehnter Band", " Band 10") \
                        .replace("Elfter Band", " Band 11").replace("Zwölfter Band", " Band 12")
                    title = title.replace(" I Band", " Band 1").replace(" II Band", " Band 2") \
                        .replace(" III Band", " Band 3").replace(" IV Band", " Band 4") \
                        .replace(" V Band", " Band 5").replace(" VI Band", " Band 6") \
                        .replace(" VII Band", " Band 7").replace(" VIII Band", " Band 8") \
                        .replace(" IX Band", " Band 9").replace(" X Band", " Band 10") \
                        .replace(" XI Band", " Band 11").replace(" XII Band", " Band 12")
                    title = title.replace("Band I ", " Band 1 ").replace("Band II ", " Band 2 ") \
                        .replace("Band III ", " Band 3 ").replace("Band IV ", " Band 4 ") \
                        .replace("Band V ", " Band 5 ").replace("Band VI ", " Band 6 ") \
                        .replace("Band VII ", " Band 7 ").replace("Band VIII ", " Band 8 ") \
                        .replace("Band IX ", " Band 9 ").replace("Band X ", " Band 10 ") \
                        .replace("Band XI ", " Band 11 ").replace("Band XII ", " Band 12 ")
                    title = title.replace(" 1 Band", " Band 1").replace(" 2 Band", " Band 2") \
                        .replace(" 3 Band", " Band 3").replace(" 4 Band", " Band 4") \
                        .replace(" 5 Band", " Band 5").replace(" 6 Band", " Band 6") \
                        .replace(" 7 Band", " Band 7").replace(" 8 Band", " Band 8") \
                        .replace(" 9 Band", " Band 9").replace(" 10 Band", " Band 10") \
                        .replace(" 11 Band", " Band 11").replace(" 12 Band", " Band 12")
                    title = title.replace("Band 1", " Band 1").replace("Band 2", " Band 2") \
                        .replace("Band 3", " Band 3").replace("Band 4", " Band 4") \
                        .replace("Band 5", " Band 5").replace("Band 6", " Band 6") \
                        .replace("Band 7", " Band 7").replace("Band 8", " Band 8") \
                        .replace("Band 9", " Band 9").replace("Band 10", " Band 10") \
                        .replace("Band 11", " Band 11").replace("Band 12", " Band 12")

                    title = title.replace("Erster Teil", " Band 1").replace("Zweiter Teil", " Band 2") \
                        .replace("Dritter Teil", " Band 3").replace("Vierter Teil", " Band 4") \
                        .replace("Fünfter Teil", " Band 5").replace("Sechster Teil", " Band 6") \
                        .replace("Siebter Teil", " Band 7").replace("Achter Teil", " Band 8") \
                        .replace("Neunter Teil", " Band 9").replace("Zehnter Teil", " Band 10") \
                        .replace("Elfter Teil", " Band 11").replace("Zwölfter Teil", " Band 12")
                    title = title.replace("1 Teil", " Band 1").replace("2 Teil", " Band 2") \
                        .replace("3 Teil", " Band 3").replace("4 Teil", " Band 4") \
                        .replace("5 Teil", " Band 5").replace("6 Teil", " Band 6") \
                        .replace("7 Teil", " Band 7").replace("8 Teil", " Band 8") \
                        .replace("9 Teil", " Band 9").replace("10 Teil", " Band 10") \
                        .replace("11 Teil", " Band 11").replace("12 Teil", " Band 12")
                    title = title.replace("Teil 1", " Band 1").replace("Teil 2", " Band 2") \
                        .replace("Teil 3", " Band 3").replace("Teil 4", " Band 4") \
                        .replace("Teil 5", " Band 5").replace("Teil 6", " Band 6") \
                        .replace("Teil 7", " Band 7").replace("Teil 8", " Band 8") \
                        .replace("Teil 9", " Band 9").replace("Teil 10", " Band 10") \
                        .replace("Teil 11", " Band 11").replace("Teil 12", " Band 12")
                    title = title.replace("Teil I ", " Band 1 ").replace("Teil II ", " Band 2 ") \
                        .replace("Teil III ", " Band 3 ").replace("Teil IV ", " Band 4 ") \
                        .replace("Teil V ", " Band 5 ").replace("Teil VI ", " Band 6 ") \
                        .replace("Teil VII ", " Band 7 ").replace("Teil VIII ", " Band 8 ") \
                        .replace("Teil IX ", " Band 9 ").replace("Teil X ", " Band 10 ") \
                        .replace("Teil XI ", " Band 11 ").replace("Teil XII ", " Band 12 ")

                    if "band" not in title.lower():
                        # print('<', title)
                        last_part_old = title.split()[-1]
                        last_part = last_part_old.replace("1", "Band 1").replace("2", "Band 2") \
                            .replace("3", "Band 3").replace("4", "Band 4") \
                            .replace("5", "Band 5").replace("6", "Band 6") \
                            .replace("7", "Band 7").replace("8", "Band 8") \
                            .replace("9", "Band 9").replace("10", "Band 10") \
                            .replace("11", "Band 11").replace("12", "Band 12")
                        title = title.replace(last_part_old, last_part)

                    title = re.sub(' +', ' ', title)
                    title = title.strip()

                    print(current_title, "|", title)
                    if current_title.replace(" ", "") == title.replace(" ", ""):
                        print("match for", doc_id)
                        nr_matches += 1
                        relevant_id = doc_id
                        break

            mapped_doc_ids[book_id] = relevant_id

    print(mapped_doc_ids)
    new_series_dict = {series_id: [mapped_doc_ids[book_id] for book_id in book_ids]
                       for series_id, book_ids in corpus.series_dict.items()}
    print(nr_matches, nr_series_books)
    big_corpus.set_series_dict(new_series_dict)
    big_corpus.save_corpus_meta("corpora/german_books")
    print(new_series_dict)


if __name__ == '__main__':
    series_dict_from_series_corpus()

    # corpus = Corpus.fast_load(path="corpora/german_series", load_entities=False)
    # big_corpus = Corpus.fast_load(path="corpora/german_books", load_entities=False)
    # c = 0
    # for doc_id, document in big_corpus.documents.items():
    #     if document.title[-1].isdigit():
    #         print(document.authors, document.title)
    #         c += 1
    # print(c)

    # corpus = DataHandler.load_real_series_books_as_corpus()
    # for i, (doc_id, doc) in enumerate(corpus.documents.items()):
    #     rev = {doc_id: series_id for series_id, doc_ids in corpus.series_dict.items() for doc_id in doc_ids}
    #     print(i, doc_id, rev[doc_id], doc)
    #
    # config = ConfigLoader.get_config()
    # annotated_corpus_path = os.path.join(config["system_storage"]["corpora"], f'{"german_big_series"}')
    # Preprocesser.annotate_and_save(corpus, corpus_dir=annotated_corpus_path, without_spacy=False)
