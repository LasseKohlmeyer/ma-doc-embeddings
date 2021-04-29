import json
import math
import os
import re
from collections import defaultdict
from enum import Enum
import random
from typing import Union, List, Dict, Tuple, Set, Generator, Any
import pandas as pd
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm
import spacy
from os import listdir
from os.path import isfile, join
import logging
import numpy as np
from aux_utils import ConfigLoader, Utils
from gutenberg_meta import load_gutenberg_meta

config = ConfigLoader.get_config()


class DataHandler:
    @staticmethod
    def build_config_str(number_of_subparts: int, size: int, dataset: str, filter_mode: str,
                         vectorization_algorithm: str, fake_series: str):
        return f'{dataset}_{number_of_subparts}_{size}_{filter_mode}_{fake_series}_{vectorization_algorithm}'

    @staticmethod
    def raw_text_parse(raw_text: str, raw_text_parse_fun):
        return raw_text_parse_fun(raw_text)

    @staticmethod
    def parse_func_german_books(raw_text: str):
        return raw_text.replace('\n', ' ').replace('  ', ' ').replace('  ', ' ')

    @staticmethod
    def parse_func_german_books_tagged(raw_text: str):
        sentences = []
        tokens = []
        for line in raw_text.split('\n'):
            try:
                token, pos, lemma = line.split('\t')
                tokens.append(Token(text=token, lemma=lemma, pos=pos))
            except ValueError:
                if line == '<SENT>':
                    sentences.append(Sentence(tokens))
                    tokens = []
                elif line == '' or line is None:
                    # skip line
                    pass
                else:
                    raise ValueError

        return ' '.join([' '.join(sentence.representation()) for sentence in sentences])

    @staticmethod
    def parse_func_dta(raw_text: str):
        raw_text = BeautifulSoup(raw_text, 'xml').select("TEI text body")[0].getText()
        return raw_text.replace("ſ", "s").replace("¬\n", "").replace("\n", " ")

    @staticmethod
    def parse_func_litrec(raw_text: str):
        content = raw_text.replace('\n@\n', ' ').replace('\n', ' ').replace('  ', ' ').replace('  ', ' ')
        return ' '.join([token.split('/')[0] for token in content.split()])

    @staticmethod
    def parse_func_goodreads(raw_text: str):
        content = raw_text.replace('\n@\n', ' ').replace('\n', ' ').replace('  ', ' ').replace('  ', ' ')
        return ' '.join([token.split('/')[0] for token in content.split()])

    @staticmethod
    def parse_func_pass(raw_text: str):
        return raw_text

    @staticmethod
    def load_corpus(input_str: str):
        if input_str == "german_books":
            return DataHandler.load_german_books_as_corpus()
        elif input_str == "tagged_german_books":
            return DataHandler.load_tagged_german_books_as_corpus()
        elif input_str == "german_series":
            return DataHandler.load_real_series_books_as_corpus()
        elif input_str == "litrec":
            return DataHandler.load_litrec_books_as_corpus()
        elif input_str == "summaries":
            return DataHandler.load_book_summaries_as_corpus()
        elif input_str == "test_corpus":
            return DataHandler.load_test_corpus()
        elif input_str == "dta":
            return DataHandler.load_dta_as_corpus()
        elif input_str == "dta_series":
            return DataHandler.load_real_series_dta_as_corpus()
        elif input_str == "goodreads_genres":
            return DataHandler.load_maharjan_goodreads()
        elif input_str == "classic_gutenberg":
            return DataHandler.load_classic_gutenberg_as_corpus()
        else:
            raise UserWarning(f"Unknown input string {input_str}!")

    @staticmethod
    def load_test_corpus():
        d0 = Document(doc_id='d_0',
                      text='Das ist das erste Dokument, erstellt am 10.10.2010 um 17 Uhr gegen Mittag in der Slovakei. '
                           'Der zweite Satz stammt aus Hessen.'
                           'Der Dritte aus dem All.',
                      title="Erstes Dokument",
                      language=Language.DE,
                      authors="LL",
                      date="2010",
                      genres=None)
        d1 = Document(doc_id='d_1',
                      text='Das ist das zweite Dokument, erstellt am 20.10.2020 um 18 Uhr am Abend von '
                           'Hans Peter Stefan in New York and Amsterdam. '
                           'Der zweite Satz kommt von niemand geringerem als Elvis Presley',
                      title="Zweites Dokument",
                      language=Language.DE,
                      authors="LK",
                      date="2020",
                      genres=None)
        d2 = Document(doc_id='d_2',
                      text='Das ist das dritte Dokument.'
                           'Es ist ganz anders als die anderen und enthält nur häusliche Informationen.'
                           'Im Haus ist es am schönsten.'
                           'Den Garten mag ich auch. Im Garten leben viele Vögel.',
                      title="Drittes Dokument",
                      language=Language.DE,
                      authors="LK",
                      date="2020",
                      genres=None)
        docs = {'d_0': d0, 'd_1': d1, 'd_2': d2}
        return Corpus(docs, language=Language.DE, name="small_test_corpus")

    @staticmethod
    def load_book_summaries_as_corpus(path: str = None) -> "Corpus":
        if path is None:
            path = config["data_set_path"]["summaries"]

        book_summary_df = pd.read_csv(path, delimiter='\t')

        # book_summary_df.columns = [['ID_A', 'ID_B', 'TITLE', 'AUTHORS', 'DATE', 'GENRES', 'TEXT']]
        # print(book_summary_df[['GENRES']].head())

        documents = {}
        for i, row in tqdm(book_summary_df.iterrows(), total=len(book_summary_df.index), desc="Parse Documents"):
            doc_id = f'bs_{i}'
            # try:
            #     genres = '--'.join(ast.literal_eval(row["GENRES"]).values())
            # except ValueError:
            #     genres = None
            genres = None
            text = DataHandler.raw_text_parse(row["TEXT"], DataHandler.parse_func_pass)
            documents[doc_id] = Document(doc_id=doc_id,
                                         text=text,
                                         title=row["TITLE"],
                                         language=Language.EN,
                                         authors=row["AUTHORS"],
                                         date=row["DATE"],
                                         genres=genres,
                                         parse_fun=None)

        return Corpus(source=documents, name="book_summaries", language=Language.EN)

    @staticmethod
    def title_replacement_ger(title: str):
        title = f'{title} '
        title = title.replace("Erster Band", "Band 1").replace("Zweiter Band", "Band 2") \
            .replace("Dritter Band", "Band 3").replace("Vierter Band", "Band 4") \
            .replace("Fünfter Band", "Band 5").replace("Sechster Band", "Band 6") \
            .replace("Siebter Band", "Band 7").replace("Achter Band", "Band 8") \
            .replace("Neunter Band", "Band 9").replace("Zehnter Band", "Band 10") \
            .replace("Elfter Band", "Band 11").replace("Zwölfter Band", "Band 12")
        title = title.replace(" I Band", " Band 1").replace(" II Band", " Band 2") \
            .replace(" III Band", " Band 3").replace(" IV Band", " Band 4") \
            .replace(" V Band", " Band 5").replace(" VI Band", " Band 6") \
            .replace(" VII Band", " Band 7").replace(" VIII Band", " Band 8") \
            .replace(" IX Band", " Band 9").replace(" X Band", " Band 10") \
            .replace(" XI Band", " Band 11").replace(" XII Band", " Band 12")
        title = title.replace("Band I ", "Band 1 ").replace("Band II ", "Band 2 ") \
            .replace("Band III ", "Band 3 ").replace("Band IV ", "Band 4 ") \
            .replace("Band V ", "Band 5 ").replace("Band VI ", "Band 6 ") \
            .replace("Band VII ", "Band 7 ").replace("Band VIII ", "Band 8 ") \
            .replace("Band IX ", "Band 9 ").replace("Band X ", "Band 10 ") \
            .replace("Band XI ", "Band 11 ").replace("Band XII ", "Band 12 ")
        title = title.replace(" 1 Band", " Band 1").replace(" 2 Band", " Band 2") \
            .replace(" 3 Band", " Band 3").replace(" 4 Band", " Band 4") \
            .replace(" 5 Band", " Band 5").replace(" 6 Band", " Band 6") \
            .replace(" 7 Band", " Band 7").replace(" 8 Band", " Band 8") \
            .replace(" 9 Band", " Band 9").replace(" 10 Band", " Band 10") \
            .replace(" 11 Band", " Band 11").replace(" 12 Band", " Band 12")

        title = title.replace("Erster Teil", "Band 1").replace("Zweiter Teil", "Band 2") \
            .replace("Dritter Teil", "Band 3").replace("Vierter Teil", "Band 4") \
            .replace("Fünfter Teil", "Band 5").replace("Sechster Teil", "Band 6") \
            .replace("Siebter Teil", "Band 7").replace("Achter Teil", "Band 8") \
            .replace("Neunter Teil", "Band 9").replace("Zehnter Teil", "Band 10") \
            .replace("Elfter Teil", "Band 11").replace("Zwölfter Teil", "Band 12")
        title = title.replace("Teil 1", "Band 1").replace("Teil 2", "Band 2") \
            .replace("Teil 3", "Band 3").replace("Teil 4", "Band 4") \
            .replace("Teil 5", "Band 5").replace("Teil 6", "Band 6") \
            .replace("Teil 7", "Band 7").replace("Teil 8", "Band 8") \
            .replace("Teil 9", "Band 9").replace("Teil 10", "Band 10") \
            .replace("Teil 11", "Band 11").replace("Teil 12", "Band 12")
        title = title.replace("1 Teil", " Band 1").replace("2 Teil", " Band 2") \
            .replace("3 Teil", " Band 3").replace("4 Teil", " Band 4") \
            .replace("5 Teil", " Band 5").replace("6 Teil", " Band 6") \
            .replace("7 Teil", " Band 7").replace("8 Teil", " Band 8") \
            .replace("9 Teil", " Band 9").replace("10 Teil", " Band 10") \
            .replace("11 Teil", " Band 11").replace("12 Teil", " Band 12")
        title = title.replace("Teil I ", "Band 1 ").replace("Teil II ", "Band 2 ") \
            .replace("Teil III ", "Band 3 ").replace("Teil IV ", "Band 4 ") \
            .replace("Teil V ", "Band 5 ").replace("Teil VI ", "Band 6 ") \
            .replace("Teil VII ", "Band 7 ").replace("Teil VIII ", "Band 8 ") \
            .replace("Teil IX ", "Band 9 ").replace("Teil X ", "Band 10 ") \
            .replace("Teil XI ", "Band 11 ").replace("Teil XII ", "Band 12 ")

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
            # print('<>', title)
        title = title.strip()
        return title


    @staticmethod
    def load_german_books_as_corpus(path: str = None) -> "Corpus":
        def load_textfile_book(prefix_path, suffix_path, document_id):
            doc_path = join(prefix_path, suffix_path)
            if not os.path.isfile(doc_path):
                raise UserWarning(f"No file found! {doc_path}")
            # with open(doc_path, "r", encoding="utf-8") as file:
            #     # content = file.read().replace('\n', ' ').replace('  ', ' ').replace('  ', ' ')
            #     # content = DataHandler.raw_text_parse(file.read(), DataHandler.parse_func_german_books)
            content = ""
            # print(content)
            meta = suffix_path.replace('.txt', '').replace('(', '').replace(')', '').split('_-_')
            author = meta[0].replace('_', ' ')
            title_year = ' '.join(meta[1:]).replace('_', ' ').replace('–', ' ').replace('.', ' ').replace(',', ' ')
            title_year = re.sub(r"\s+", " ", title_year)
            title = title_year[:-4]

            try:
                year = int(title_year[-4:])
            except ValueError:
                title = title_year
                year = None

            title = DataHandler.title_replacement_ger(title)
            # print(author, '|', title, '|', year)
            d = Document(doc_id=document_id,
                         text=content,
                         title=title,
                         language=Language.DE,
                         authors=author,
                         date=str(year),
                         genres=None,
                         parse_fun=DataHandler.parse_func_german_books,
                         file_path=doc_path)
            return d

        if path is None:
            path = config["data_set_path"]["german_books"]

        path_a = join(path, 'corpus-of-german-fiction-txt')
        path_b = join(path, 'corpus-of-translated-foreign-language-fiction-txt')
        german_fiction = [f for f in listdir(path_a) if isfile(join(path_a, f))]
        tanslated_fiction = [f for f in listdir(path_b) if isfile(join(path_b, f))]

        documents = {}
        for i, german_fiction_path in enumerate(german_fiction):
            doc_id = f'gfo_{i}'
            documents[doc_id] = load_textfile_book(path_a, german_fiction_path, doc_id)

        for i, translated_fiction_path in enumerate(tanslated_fiction):
            doc_id = f'gft_{i}'
            documents[doc_id] = load_textfile_book(path_b, translated_fiction_path, doc_id)

        return Corpus(source=documents, name="german_fiction", language=Language.DE)

    @staticmethod
    def load_real_series_books_as_corpus(path: str = None) -> "Corpus":
        def load_textfile_book(prefix_path, suffix_path, document_id):
            doc_path = join(prefix_path, suffix_path)
            if not os.path.isfile(doc_path):
                raise UserWarning(f"No file found! {doc_path}")
            # with open(doc_path, "r", encoding="utf-8") as file:
            #     # content = file.read().replace('\n', ' ').replace('  ', ' ').replace('  ', ' ')
            #     # content = DataHandler.raw_text_parse(file.read(), DataHandler.parse_func_german_books)
            content = ""
            # print(content)
            meta = suffix_path.replace('.txt', '').replace('(', '').replace(')', '').split('_-_')
            author = meta[0].replace('_', ' ')
            title_year = ' '.join(meta[1:]).replace('_', ' ').replace('–', ' ').replace('.', ' ').replace(',', ' ')
            title_year = re.sub(r"\s+", " ", title_year)
            title = title_year[:-4]
            # print(title)
            try:
                year = int(title_year[-4:])
            except ValueError:
                title = title_year
                year = None

            title = DataHandler.title_replacement_ger(title)

            # print(author, '|', title, '|', year)
            d = Document(doc_id=document_id,
                         text=content,
                         title=title,
                         language=Language.DE,
                         authors=author,
                         date=str(year),
                         genres=None,
                         parse_fun=DataHandler.parse_func_german_books,
                         file_path=doc_path)
            return d

        if path is None:
            path = config["data_set_path"]["german_books"]

        path_a = join(path, 'corpus-of-german-fiction-txt')
        path_b = join(path, 'corpus-of-translated-foreign-language-fiction-txt')
        german_fiction = [f for f in listdir(path_a) if isfile(join(path_a, f))]
        tanslated_fiction = [f for f in listdir(path_b) if isfile(join(path_b, f))]
        regexp = re.compile(r'_\d+_?')
        series_paths = [(path_a, path) for path in german_fiction
                        # if "band" in path.lower() or "teil" in path.lower() or regexp.search(path)
                        ]
        series_paths.extend([(path_b, path) for path in tanslated_fiction
                             # if "band" in path.lower() or "teil" in path.lower() or regexp.search(path)
                             ])

        documents = {}
        for i, path in enumerate(series_paths):
            p_dir, p_file = path
            doc_id = f'sgf_{i}'
            documents[doc_id] = load_textfile_book(p_dir, p_file, doc_id)

        # print(documents)

        suffix_dict = defaultdict(list)
        for doc_id, document in documents.items():
            splitted_title = document.title.split('Band')
            band_title = splitted_title[0]
            if band_title == "Robin der Rote der ":
                band_title = "Robin der Rote "
            suffix_dict[(band_title.strip(), document.authors.strip())].append(doc_id)

        # print({series.strip(): doc_ids for series, doc_ids in suffix_dict.items() if len(doc_ids) > 1})
        series_dict = {series[0]: doc_ids for series, doc_ids in suffix_dict.items() if len(doc_ids) > 1}
        # for series, docs in series_dict.items():
        #     print(series, docs)
        # print(len(series_dict))

        relevant_ids = [doc_id for series, doc_ids in series_dict.items() for doc_id in doc_ids]
        documents = {doc_id: document for doc_id, document in documents.items() if doc_id in relevant_ids}

        series_documents = {}
        new_series_dict = defaultdict(list)
        for index, (series, doc_ids) in enumerate(series_dict.items()):
            try:
                for doc_id in doc_ids:
                    series_doc: Document = documents[doc_id]
                    # print(series, doc_id, doc_ids, series_doc.title)
                    series_id = int(series_doc.title.split()[-1]) - 1

                    new_doc_id = f'gs_{index}_{series_id}'
                    # print(series_doc)
                    series_doc.doc_id = new_doc_id
                    series_doc.title = series_doc.title.strip()
                    series_documents[new_doc_id] = series_doc
                    new_series_dict[f'gs_{index}'].append(new_doc_id)

            except ValueError:
                for j, doc_id in enumerate(doc_ids):
                    series_doc: Document = documents[doc_id]
                    # print(series, doc_id, doc_ids, series_doc.title)
                    series_id = j

                    new_doc_id = f'gs_{index}_{series_id}'
                    # print(series_doc)
                    series_doc.doc_id = new_doc_id
                    series_doc.title = series_doc.title.strip()
                    series_documents[new_doc_id] = series_doc
                    new_series_dict[f'gs_{index}'].append(new_doc_id)
                    # print(f"Used sub series_id {j} for {series_doc.title}")

        corpus = Corpus(source=series_documents, name="german_series", language=Language.DE)
        corpus.set_series_dict(new_series_dict)

        return corpus

    @staticmethod
    def load_dta_as_corpus(path: str = None) -> "Corpus":
        if path is None:
            input_dir = config["data_set_path"]["dta"]
        else:
            input_dir = path
        del path
        path_names = [text_file for text_file in os.listdir(input_dir) if text_file.endswith('.xml')]

        path_names = [path_name for path_name in path_names if int(path_name.split('.')[0].split('_')[-1]) >= 1800]
        d = defaultdict(list)
        for path_name in path_names:
            meta_info = path_name.split('.')[0].split('_')
            author_name = meta_info[0]
            title = meta_info[1]
            d[f'{author_name}_{title[:-2]}'].append(path_name)

        d = {key: path_names for key, path_names in d.items() if len(path_names) > 0}
        series_number = 0
        documents = {}
        series_dict = defaultdict(list)
        for series, paths in d.items():
            for inner_series_nr, path in enumerate(paths):
                doc_path = join(input_dir, path)
                with open(doc_path, 'r', encoding="utf-8") as tei:
                    tei_content = tei.read()

                    soup = BeautifulSoup(tei_content, 'xml')
                    doc_title = []
                    for title in soup.find_all('title'):
                        parsed_title = title.getText()
                        if parsed_title not in doc_title:
                            doc_title.append(parsed_title)

                    doc_title = ' '.join(list(doc_title))
                    if len(doc_title) > 40:
                        doc_title = ' '.join(doc_title.split()[:20])
                    print(doc_title)
                    authors = []
                    for author in soup.find_all('author'):
                        try:
                            parsed_author = f'{author.forename.getText()} {author.surname.getText()}'

                        except AttributeError:
                            parsed_author = f'{author.surname.getText()}'
                        if parsed_author not in authors:
                            authors.append(parsed_author)

                    doc_authors = ', '.join(authors)
                    # doc_text = soup.select("TEI text body")[0].getText()
                    # doc_text = doc_text.replace("ſ", "s").replace("¬\n", "").replace("\n", " ")

                    # doc_text = DataHandler.raw_text_parse(tei.read(), DataHandler.parse_func_dta)
                    doc_text = ""

                    doc_date = path.split('.')[0].split('_')[-1]
                    doc_id = f"dta_{series_number}_{inner_series_nr}"
                    # print(doc_id, path, doc_date, doc_title, doc_authors)
                    # print(doc_text[:100], '---', doc_text[-100:])
                    doc = Document(doc_id=doc_id,
                                   text=doc_text,
                                   title=doc_title,
                                   language=Language.DE,
                                   authors=doc_authors,
                                   date=doc_date,
                                   genres=None,
                                   sentences=None,
                                   parse_fun=DataHandler.parse_func_dta,
                                   file_path=doc_path)

                    documents[doc_id] = doc
                    series_dict[f"dta_{series_number}"].append(doc_id)
            series_number += 1
        series_dict = {series_id: doc_ids for series_id, doc_ids in series_dict.items() if len(doc_ids) > 1}
        corpus = Corpus(documents, name="deutsches_text_archiv_belletristik_series", language=Language.DE)
        corpus.set_series_dict(series_dict)
        return corpus

    @staticmethod
    def load_real_series_dta_as_corpus(path: str = None):
        if path is None:
            input_dir = config["data_set_path"]["dta"]
        else:
            input_dir = path
        del path
        path_names = [text_file for text_file in os.listdir(input_dir) if text_file.endswith('.xml')]

        path_names = [path_name for path_name in path_names if int(path_name.split('.')[0].split('_')[-1]) > 1800]
        d = defaultdict(list)
        for path_name in path_names:
            meta_info = path_name.split('.')[0].split('_')
            author_name = meta_info[0]
            title = meta_info[1]
            d[f'{author_name}_{title[:-2]}'].append(path_name)

        d = {key: path_names for key, path_names in d.items() if len(path_names) > 1}
        series_number = 0
        documents = {}
        series_dict = defaultdict(list)
        for series, paths in d.items():
            for inner_series_nr, path in enumerate(paths):
                doc_path = join(input_dir, path)
                with open(doc_path, 'r', encoding="utf-8") as tei:
                    tei_content = tei.read()

                    soup = BeautifulSoup(tei_content, 'xml')
                    doc_title = []
                    for title in soup.find_all('title'):
                        parsed_title = title.getText()
                        if parsed_title not in doc_title:
                            doc_title.append(parsed_title)
                    doc_title = ' '.join(list(doc_title))

                    authors = []
                    for author in soup.find_all('author'):

                        try:
                            parsed_author = f'{author.forename.getText()} {author.surname.getText()}'

                        except AttributeError:
                            parsed_author = f'{author.surname.getText()}'
                        if parsed_author not in authors:
                            authors.append(parsed_author)

                    doc_authors = ', '.join(authors)
                    # doc_text = soup.select("TEI text body")[0].getText()
                    # doc_text = doc_text.replace("ſ", "s").replace("¬\n", "").replace("\n", " ")

                    # doc_text = DataHandler.raw_text_parse(tei.read(), DataHandler.parse_func_dta)
                    doc_text = ""

                    doc_date = path.split('.')[0].split('_')[-1]
                    doc_id = f"dta_{series_number}_{inner_series_nr}"
                    # print(doc_id, path, doc_date, doc_title, doc_authors)
                    # print(doc_text[:100], '---', doc_text[-100:])
                    doc = Document(doc_id=doc_id,
                                   text=doc_text,
                                   title=doc_title,
                                   language=Language.DE,
                                   authors=doc_authors,
                                   date=doc_date,
                                   genres=None,
                                   sentences=None,
                                   parse_fun=DataHandler.parse_func_dta,
                                   file_path=doc_path)

                    documents[doc_id] = doc
                    series_dict[f"dta_{series_number}"].append(doc_id)
            series_number += 1
        corpus = Corpus(documents, name="deutsches_text_archiv_belletristik_series", language=Language.DE)
        corpus.set_series_dict(series_dict)
        return corpus

    @staticmethod
    def load_tagged_german_books_as_corpus(path: str = None) -> "Corpus":
        def load_textfile_book(prefix_path, suffix_path, document_id):
            doc_path = join(prefix_path, suffix_path)
            if not os.path.isfile(doc_path):
                raise UserWarning(f"No file found! {doc_path}")
            # with open(doc_path, "r", encoding="utf-8") as file:
            #     # content = DataHandler.raw_text_parse(file.read(), DataHandler.parse_func_german_books_tagged)
            content = ""
            # print(content)
            meta = suffix_path.replace('.tagged.corr.tsv', '').split('_-_')
            year_author = meta[0].split('_')
            year = int(year_author[0])
            author = ' '.join(year_author[1:])
            title = ' '.join(meta[1:]).replace('_', ' ')
            # print(author, '|', title, '|', year)
            d = Document(doc_id=document_id,
                         text=content,
                         title=title,
                         language=Language.DE,
                         authors=author,
                         date=str(year),
                         genres=None,
                         parse_fun=DataHandler.parse_func_german_books_tagged,
                         file_path=doc_path)
            # d.set_sentences(sentences)
            return d

        if path is None:
            path = config["data_set_path"]["tagged_german_books"]

        german_fiction = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.tsv')]

        documents = {}
        for i, german_fiction_path in tqdm(enumerate(german_fiction), total=len(german_fiction)):
            doc_id = f'tgfo_{i}'
            documents[doc_id] = load_textfile_book(path, german_fiction_path, doc_id)

        return Corpus(source=documents, name="german_fiction_tagged", language=Language.DE)

    @staticmethod
    def load_classic_gutenberg_as_corpus(path: str = None) -> "Corpus":
        def load_textfile_book(prefix_path, suffix_path, document_id):
            doc_path = join(prefix_path, suffix_path)
            if not os.path.isfile(doc_path):
                raise UserWarning(f"No file found! {doc_path}")
            # with open(doc_path, "r", encoding="utf-8") as file:
            #     # content = file.read().replace('\n', ' ').replace('  ', ' ').replace('  ', ' ')
            #     content = DataHandler.raw_text_parse(file.read(), DataHandler.parse_func_german_books)
            content = ""

            meta = suffix_path.replace('.txt', '').split('_-_')
            author = meta[1].replace('_', ' ')
            year = meta[2]
            title = meta[0].replace('_', ' ')

            print(author, '|', title, '|', year)
            d = Document(doc_id=document_id,
                         text=content,
                         title=title,
                         language=Language.DE,
                         authors=author,
                         date=str(year),
                         genres=None,
                         parse_fun=DataHandler.parse_func_german_books,
                         file_path=doc_path)
            return d

        if path is None:
            path = config["data_set_path"]["gutenberg_top_20"]

        file_paths = [f for f in listdir(path) if isfile(join(path, f))]

        documents = {}
        for i, file_path in enumerate(file_paths):
            doc_id = f'cb_{i}'
            documents[doc_id] = load_textfile_book(path, file_path, doc_id)

        return Corpus(source=documents, name="classic_books", language=Language.EN)

    @staticmethod
    def load_litrec_books_as_corpus(corpus_dir: str = None) -> "Corpus":
        def load_textfile_book(prefix_path, suffix_path, document_id, title):
            doc_path = join(prefix_path, suffix_path)
            if not os.path.isfile(doc_path):
                raise FileNotFoundError  # UserWarning(f"No file found! {doc_path}")
            # with open(doc_path, "r", encoding="utf-8") as file:
            #     # content = file.read().replace('\n@\n', ' ').replace('\n', ' ').replace('  ', ' ').replace('  ', ' ')
            #     # content = ' '.join([token.split('/')[0] for token in content.split()])
            #     # content = DataHandler.raw_text_parse(file.read(), DataHandler.parse_func_litrec)
            content = ""

            meta = suffix_path.replace('a.txt.clean.pos', '').split('-')
            author = meta[0].replace('+', ' ')
            year = None
            # print(author, '|', title, '|', year)
            d = Document(doc_id=document_id,
                         text=content,
                         title=title,
                         language=Language.DE,
                         authors=author,
                         date=str(year),
                         genres=None,
                         parse_fun=DataHandler.parse_func_litrec,
                         file_path=doc_path)
            return d

        if corpus_dir is None:
            corpus_dir = config["data_set_path"]["litrec"]
        df = pd.read_csv(join(corpus_dir, 'user-ratings-v1.txt'), delimiter='@')

        documents = {}
        not_found = []

        filenames = df[['filename', 'title']].drop_duplicates()
        for i, row in tqdm(filenames.iterrows(), total=len(filenames.index), disable=True):
            doc_id = f'lr_{i}'
            try:
                documents[doc_id] = load_textfile_book(prefix_path=join(corpus_dir, 'books-v11'),
                                                       suffix_path=row['filename'],
                                                       document_id=doc_id,
                                                       title=row['title'])
            except FileNotFoundError:
                not_found.append((row['title'], row['filename']))
        # print(len(not_found))
        return Corpus(source=documents, name="litrec", language=Language.EN)


    @staticmethod
    def load_maharjan_goodreads(corpus_dir: str = None) -> "Corpus":
        def load_textfile_book(prefix_path, text_genre, success_status, suffix_path, document_id, ):
            doc_path = join(prefix_path, text_genre, success_status, suffix_path)
            if not os.path.isfile(doc_path):
                raise UserWarning(f"No file found! {doc_path}")

            # with open(doc_path, "r", encoding="utf-8") as file:
            # base_content = file.read()
            # content = base_content.replace('\n@\n', ' ').replace('\n', ' ').replace('  ', ' ') \
            #     .replace('  ', ' ')
            # content = ' '.join([token.split('/')[0] for token in content.split()])
            # content = DataHandler.raw_text_parse(file.read(), DataHandler.parse_func_goodreads)

            file_path_splitted = suffix_path.replace('.txt', '').split('_')
            guten_id = file_path_splitted[0]
            title = file_path_splitted[-1].replace('+', ' ').title()

            content = ""
            year = None
            # meta_data_range = '\n'.join(base_content.split('\n')[:30])
            # year_matches = re.findall(r'([1-2][0-9]{3})', meta_data_range)
            # if year_matches:
            #     year = year_matches[0]

            try:
                author = guten_dict[str(guten_id)][1]
            except KeyError:
                author = None

            # print(genre, status, title, year, author)
            # author = meta[0].replace('+', ' ')
            # print(author, '|', title, '|', year)
            d = Document(doc_id=document_id,
                         text=content,
                         title=title,
                         language=Language.DE,
                         authors=author,
                         date=str(year),
                         genres=text_genre,
                         parse_fun=DataHandler.parse_func_goodreads,
                         file_path=doc_path)
            return d

        if corpus_dir is None:
            corpus_dir = config["data_set_path"]["maharjan_goodreads"]

        guten_dict = load_gutenberg_meta(config["data_set_path"]["gutenberg_meta"])

        genres = [genre_dir for genre_dir in listdir(corpus_dir) if os.path.isdir(join(corpus_dir, genre_dir))
                  if genre_dir != "dismissed"]
        genres_dict = {}
        for genre in genres:

            load_genre = {
                "failure": [f for f in listdir(os.path.join(corpus_dir, genre, "failure"))
                            if isfile(join(corpus_dir, genre, "failure", f))],
                "success": [f for f in listdir(os.path.join(corpus_dir, genre, "success"))
                            if isfile(join(corpus_dir, genre, "success", f))]
            }
            genres_dict[genre] = load_genre


        documents = {}
        succes_dict = {}
        book_counter = 0
        for genre, genre_dict in genres_dict.items():
            books = [(book, "failure") for book in genre_dict["failure"]]
            books.extend([(book, "success") for book in genre_dict["success"]])

            for (book, status) in books:
                doc_id = f"gr_{book_counter}"
                documents[doc_id] = load_textfile_book(corpus_dir, genre, status, book, doc_id)
                succes_dict[doc_id] = status
                book_counter += 1
        corpus = Corpus(source=documents, name="goodreads", language=Language.EN)
        corpus.success_dict = succes_dict

        return corpus


class Language(str, Enum):
    UNKNOWN = "unknown"
    DE = "de"
    EN = "en"

    @staticmethod
    def get_from_str(language: str) -> "Language":
        if language.lower() == "en" or language.lower() == "english" or \
                language.lower() == "englisch":
            return Language.EN
        if language.lower() == "de" or language.lower() == "deutsch" or \
                language.lower() == "ger" or language.lower() == "german":
            return Language.DE
        return Language.UNKNOWN


def clean_token(token: str):
    sub = re.sub('[^A-Za-z0-9.,?!]+', '', token)
    if sub == "":
        sub = ","
    return sub


class Token:
    __slots__ = 'text', 'lemma', 'pos', 'ne', 'punctuation', 'alpha', 'stop'

    def __init__(self, text: str,
                 lemma: str = None,
                 pos: str = None,
                 ne: str = None,
                 punctuation: bool = None,
                 alpha: bool = None,
                 stop: bool = None):
        self.text = text
        self.lemma = lemma
        self.pos = pos
        self.ne = ne
        self.punctuation = punctuation
        self.alpha = alpha
        self.stop = stop

    def representation(self, lemma: bool = False, lower: bool = False, token_retrieve: bool = False):
        if token_retrieve:
            return self
        if lemma:
            rep = self.lemma
        else:
            rep = self.text
        if lower:
            rep = rep.lower()
        rep = clean_token(rep)
        return rep

    def json_representation(self):
        return vars(self)

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and
                self.text == other.text and
                self.lemma == other.lemma and
                self.pos == other.pos and
                self.ne == other.ne and
                self.punctuation == other.punctuation and
                self.alpha == other.alpha and
                self.stop == other.stop)

    def __hash__(self):
        return (hash(self.text) + hash(self.lemma) + hash(self.pos) + hash(self.ne) + hash(self.text)
                + hash(self.punctuation) + hash(self.alpha) + hash(self.stop))

    def __repr__(self):
        return f'|{self.text}|'

    def get_save_file_representation(self, flair_mode: str = None):
        def bool_converter(input_bool: bool) -> str:
            if input_bool:
                return "1"
            else:
                return "0"
        if flair_mode:
            return f'{self.text}\t{self.lemma}\t{str(self.pos).strip()}\t{str(self.ne).strip()}' \
                   f'\t{bool_converter(self.punctuation)}' \
                   f'\t{bool_converter(self.alpha)}\t{bool_converter(self.stop)}\t{flair_mode}'
        else:
            return f'{self.text}\t{self.lemma}\t{str(self.pos).strip()}\t{str(self.ne).strip()}' \
                   f'\t{bool_converter(self.punctuation)}' \
                   f'\t{bool_converter(self.alpha)}\t{bool_converter(self.stop)}'

    @staticmethod
    def parse_text_file_token_representation(input_repr) -> "Token":
        def bool_unconverter(input_bool: str) -> bool:
            if input_bool == "1":
                return True
            else:
                return False

        text, lemma, pos, ne, punctuation, alpha, stop = input_repr.split('\t')
        punctuation = bool_unconverter(punctuation)
        alpha = bool_unconverter(alpha)
        stop = bool_unconverter(stop)

        return Token(text=text, lemma=lemma, pos=pos, ne=ne, punctuation=punctuation, alpha=alpha, stop=stop)

    @classmethod
    def empty_token(cls):
        return Token(text="del", lemma="del", pos=None, ne=None, punctuation=None, alpha=None, stop=None)

    # __repr__ = __str__


class Sentence:
    __slots__ = 'tokens', "length"

    # def __init__(self, tokens: List[str]):
    #     self.tokens = tokens
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.length = sum((1 for token in self.tokens if token.text != "del"))

    def representation(self, lemma: bool = False, lower: bool = False):
        return [token.representation(lemma, lower) for token in self.tokens]

    def json_representation(self):
        return vars(self)

    def __str__(self):
        return str(self.representation())

    __repr__ = __str__

    def truncate(self, n: int):
        self.tokens = self.tokens[:n]
        self.length = sum((1 for token in self.tokens if token.text != "del"))

    def __len__(self):
        return self.length


class Document:
    __slots__ = 'doc_id', 'text', 'title', 'language', 'authors', 'date', 'genres', 'sentences', \
                'absolute_positions', 'file_path', 'length', 'vocab_size', 'sentences_nr', 'parse_fun', 'doc_entities'

    def __init__(self, doc_id: str, text: str, title: str, language: Language,
                 authors: str = None, date: str = None, genres: str = None, sentences: List[Sentence] = None,
                 file_path: str = None, length: int = 0, vocab_size: int = 0, sentence_nr: int = 0, parse_fun=None):
        self.doc_id = doc_id
        self.text = text
        self.title = title
        self.language = language
        self.authors = authors
        self.date = date
        self.genres = genres

        # self.sentences: List[Sentence] = sentences  # None
        self.absolute_positions = {}
        self.file_path = file_path

        self.sentences = None
        self.doc_entities = None
        self.length = int(length)
        self.vocab_size = int(vocab_size)
        self.sentences_nr = int(sentence_nr)
        self.set_sentences(sentences)
        self.parse_fun = parse_fun

        # self.tokens: List[str] = []  # None

    # def set_sentences(self, sentences: List[List[str]]):
    #     self.sentences = [Sentence(sentence) for sentence in sentences]
    #     self.tokens = [token for sentence in sentences for token in sentence]

    def set_sentences_from_gen(self):
        self.sentences = list(self.sentences)

    def set_sentences(self, sentences: List[Sentence]):
        if sentences is None:
            sentences = []
        self.sentences = sentences

        if len(self.sentences) != 0:
            self.length = sum((len(s) for s in self.sentences))
            self.vocab_size = len(set([t for s in self.sentences for t in s.tokens]))
            self.sentences_nr = len(self.sentences)

        # self.tokens = [token for sentence in sentences for token in sentence.tokens]

    def set_entities(self):
        doc_entities = defaultdict(list)
        if not isinstance(self.sentences, Generator) and len(self.sentences) == 0:
            sentences = self.get_sentences_from_disk()
        else:
            sentences = self.sentences

        for sent_id, sent in enumerate(sentences):
            for token_id, token in enumerate(sent.tokens):
                if token.ne:
                    # print(token.ne, token.text)
                    doc_entities[token.ne].append((sent_id, token_id, token))
        self.doc_entities = doc_entities

    def reset_text_based_on_sentences(self):
        self.text = ' '.join([' '.join(sentence.representation()) for sentence in self.sentences])

    def build_position_indices(self):
        c = 0
        for i, sentence in enumerate(self.sentences):
            for j, token in enumerate(sentence.tokens):
                self.absolute_positions[c] = (i, j)
                c += 1

    def get_token_at_doc_position(self, position: int):
        if len(self.absolute_positions) == 0:
            self.build_position_indices()
        try:
            sentence_id, token_id = self.absolute_positions[position]
        except KeyError:
            return None
        return self.sentences[sentence_id].tokens[token_id]

    def get_flat_document_tokens(self, lemma: bool = False, lower: bool = False, through_error: bool = True):
        # for doc_id, document in self.documents.items():
        #     print(document.sentences[1].tokens)
        # print('>', self.sentences)
        tokens = [token.representation(lemma, lower) for sentence in self.sentences for token in sentence.tokens
                  if token.representation(lemma=False, lower=True) != 'del']
        if len(tokens) == 0:
            if through_error:
                raise UserWarning("No sentences set")
        return tokens

    def get_flat_tokens_from_disk(self, as_list: bool = True, lemma: bool = False, lower: bool = False) -> List[str]:
        if self.file_path is None:
            raise UserWarning(f"No filepath associated with Document {self.doc_id}")
        return [token.representation(lemma, lower)
                for sentence in Document.sentences_from_doc_file(self.file_path, as_list=as_list)
                for token in sentence.tokens if token.representation(lemma=False, lower=True) != 'del']

    def get_flat_and_lda_filtered_tokens(self, lemma: bool = False, lower: bool = False, from_disk: bool = True):
        if from_disk:
            sentences = self.get_sentences_from_disk()
        else:
            sentences = self.sentences
        tokens = [token.representation(lemma, lower)
                  for sentence in sentences
                  for token in sentence.tokens
                  if not token.stop and not token.punctuation and token.text != "del"]
        if len(tokens) == 0:
            raise UserWarning("No sentences set")

        return tokens

    def get_flat_and_filtered_tokens_from_disk(self, lemma: bool = False, lower: bool = False,
                                               pos: list = None,
                                               focus_stopwords: bool = False,
                                               focus_punctuation: bool = False,
                                               focus_ne: bool = False,
                                               masking: bool = False,
                                               revert: bool = False):
        def filter_condition(token: Token):
            if revert:
                return (not focus_stopwords or not token.stop) \
                       and (not focus_punctuation or not token.alpha) \
                       and (not pos or token.pos not in pos) \
                       and (not focus_ne or not token.ne)
            else:
                return (not focus_stopwords or token.stop) \
                       and (not focus_punctuation or token.alpha) \
                       and (not pos or token.pos in pos) \
                       and (not focus_ne or token.ne)

        def mask(input_token: Token):
            output_token = Token(text=input_token.text,
                                 lemma=input_token.lemma,
                                 pos=input_token.pos,
                                 ne=input_token.ne,
                                 punctuation=input_token.punctuation,
                                 alpha=input_token.alpha,
                                 stop=input_token.stop)
            if not filter_condition(output_token):
                output_token.text = "del"
                output_token.lemma = "del"
            return output_token

        # for doc_id, document in self.documents.items():
        #     print(document.sentences[1].tokens)
        if not masking:
            tokens = [token.representation(lemma, lower)
                      for sentence in self.get_sentences_from_disk()
                      for token in sentence.tokens
                      if filter_condition(token)]
        else:
            tokens = [mask(token).representation(lemma, lower)
                      for sentence in self.get_sentences_from_disk()
                      for token in sentence.tokens]

        if len(tokens) == 0:
            raise UserWarning("No sentences set")

        return tokens

    def get_flat_and_filtered_document_tokens(self, lemma: bool = False, lower: bool = False, pos: list = None,
                                              focus_stopwords: bool = False,
                                              focus_punctuation: bool = False,
                                              focus_ne: bool = False,
                                              masking: bool = False,
                                              revert: bool = False,
                                              ids: bool = False) -> Union[List[str], List[Tuple[str, int]]]:
        def filter_condition(token: Token):
            if revert:
                return (not focus_stopwords or not token.stop) \
                       and (not focus_punctuation or not token.alpha) \
                       and (not pos or token.pos not in pos) \
                       and (not focus_ne or not token.ne)
            else:
                return (not focus_stopwords or token.stop) \
                       and (not focus_punctuation or token.alpha) \
                       and (not pos or token.pos in pos) \
                       and (not focus_ne or token.ne)

        def mask(input_token: Token):
            output_token = Token(text=input_token.text,
                                 lemma=input_token.lemma,
                                 pos=input_token.pos,
                                 ne=input_token.ne,
                                 punctuation=input_token.punctuation,
                                 alpha=input_token.alpha,
                                 stop=input_token.stop)
            if not filter_condition(output_token):
                output_token.text = "del"
                output_token.lemma = "del"
            return output_token

        # for doc_id, document in self.documents.items():
        #     print(document.sentences[1].tokens)
        if not masking:
            if ids:
                tokens = [(sentence_id, token_id)
                          for sentence_id, sentence in enumerate(self.sentences)
                          for token_id, token in enumerate(sentence.tokens)
                          if filter_condition(token)]
            else:
                tokens = [token.representation(lemma, lower)
                          for sentence in self.sentences
                          for token in sentence.tokens
                          if filter_condition(token)]
        else:
            tokens = [mask(token).representation(lemma, lower)
                      for sentence in self.sentences
                      for token in sentence.tokens]

        if len(tokens) == 0:
            raise UserWarning("No sentences set")

        return tokens

    def load_sentences_from_disk(self, as_list: bool = True):
        self.sentences = Document.sentences_from_doc_file(self.file_path, as_list=as_list)

    def get_sentences_from_disk(self, as_list: bool = True) -> List[Sentence]:
        return Document.sentences_from_doc_file(self.file_path, as_list=as_list)

    def get_text_from_disk(self):
        if self.parse_fun is None:
            raise UserWarning("No parsing function defined for document!")
        if self.file_path is None:
            raise UserWarning("No base file path defined for document!")
        return Document.text_from_doc_file(self.file_path, parse_fun=self.parse_fun)

    def get_vocab(self, from_disk: bool = True, lemma: bool = False, lower: bool = False,
                  remove_stopwords: bool = False, remove_punctuation: bool = False, lda_mode: bool = True) -> Set[str]:
        if lda_mode:
            return set(self.get_flat_and_lda_filtered_tokens(lemma=lemma, lower=lower, from_disk=from_disk))
        if from_disk:
            if remove_stopwords or remove_punctuation:
                return set(self.get_flat_and_filtered_tokens_from_disk(lemma=lemma, lower=lower,
                                                                       focus_punctuation=remove_punctuation,
                                                                       focus_stopwords=remove_stopwords,
                                                                       revert=False))
            else:
                return set(self.get_flat_tokens_from_disk(lemma=lemma, lower=lower))
        else:
            if remove_stopwords or remove_punctuation:
                return set(self.get_flat_and_filtered_document_tokens(lemma=lemma, lower=lower,
                                                                      focus_punctuation=remove_punctuation,
                                                                      focus_stopwords=remove_stopwords,
                                                                      revert=False))
            else:
                return set(self.get_flat_document_tokens(lemma=lemma, lower=lower))

    def __str__(self):
        return f'{self.authors} ({self.date}): {self.title[:50]}'

    __repr__ = __str__

    def __len__(self):
        return self.length  # sum((len(sentence) for sentence in self.sentences))

    def calculate_sizes(self, from_file: bool = False, through_error: bool = True):
        self.sentences_nr = len(self.sentences)
        if from_file:
            tokens = self.get_flat_tokens_from_disk()
        else:
            tokens = self.get_flat_document_tokens(through_error=through_error)
        self.length = len(tokens)
        self.vocab_size = len(set(tokens))

    def json_representation(self):
        return vars(self)

    def meta_string_representation_wo_length(self) -> str:
        pattern = re.compile(r'[\W]+', re.UNICODE)
        resu = f'{self.doc_id}_-_{str(self.authors).replace(" ", "_")}_-_' \
               f'{pattern.sub("", str(self.title)).replace(" ", "_")}_-_' \
               f'{self.language}_-_{str(self.genres).replace(" ", "_")}_-_{self.date}_-_'

        resu = resu.replace('"', '')
        return resu

    def meta_string_representation(self) -> str:
        pattern = re.compile(r'[\W]+', re.UNICODE)
        resu = f'{self.doc_id}_-_{str(self.authors).replace(" ", "_")}_-_' \
               f'{pattern.sub("_", str(self.title).strip())}_-_' \
               f'{self.language}_-_{str(self.genres).replace(" ", "_")}_-_{self.date}_-_' \
               f'{str(self.length)}_-_{str(self.vocab_size)}_-_{self.sentences_nr}'

        resu = resu.replace('"', '')
        return resu

    def store_to_corpus_file(self, corpus_dir: str):
        doc_path = os.path.join(corpus_dir, f'{self.meta_string_representation()}.txt')
        if not os.path.isdir(corpus_dir):
            os.mkdir(corpus_dir)
        with open(doc_path, 'w', encoding="utf-8") as writer:
            for sentence in self.sentences:
                for token in sentence.tokens:
                    writer.write(f'{token.get_save_file_representation()}\n')
                writer.write("<SENT>\n")
        return doc_path

    def check_if_doc_at_path(self, corpus_dir: str):
        files = [f for f in listdir(corpus_dir) if isfile(join(corpus_dir, f))]

        doc_path = f'{self.meta_string_representation_wo_length()}'
        for file in files:
            if file.startswith(doc_path):
                return os.path.join(corpus_dir, file)
        return None

    @staticmethod
    def sentences_from_doc_file(doc_path: str, as_list: bool = True):
        def parse_sentence(sentence_string):
            return Sentence([Token.parse_text_file_token_representation(token_ln)
                             for token_ln in sentence_string.split('\n')
                             if token_ln != '' and token_ln is not None and token_ln != '\n'])

        with open(doc_path, "r", encoding="utf-8") as file:
            if as_list:
                sentences = [parse_sentence(sentence) for sentence in file.read().split('<SENT>')]
                sentences = [sentence for sentence in sentences if len(sentence) > 0]
            else:
                sentences = (parse_sentence(sentence) for sentence in file.read().split('<SENT>'))
                sentences = (sentence for sentence in sentences if len(sentence) > 0)
        return sentences

    @staticmethod
    def text_from_doc_file(doc_path: str, parse_fun):
        with open(doc_path, "r", encoding="utf-8") as file:
            text = DataHandler.raw_text_parse(file.read(), parse_fun)
        return text

    @staticmethod
    def create_document_from_doc_file(doc_path: str, disable_sentences: bool = False):
        fn = os.path.basename(doc_path)
        splitted_meta = fn.replace('.txt', '').split('_-_')

        if len(splitted_meta) == 6:
            doc_id, authors, title, language, genres, date = splitted_meta
            length, vocab_size, sentence_nr = 0, 0, 0
        elif len(splitted_meta) == 9:
            doc_id, authors, title, language, genres, date, length, vocab_size, sentence_nr = splitted_meta
        else:
            raise UserWarning("Meta data contains neither 6 nor 9 attributes")

        title = title.replace('_', ' ')
        authors = authors.replace('_', ' ')
        genres = genres.replace('_', ' ')
        if doc_id == "None":
            doc_id = None
        if authors == "None":
            authors = None
        if title == "None":
            title = None
        if language == "None":
            language = None
        if genres == "None":
            genres = None
        if date == "None":
            date = None

        if disable_sentences:
            sentences = []
        else:
            sentences = Document.sentences_from_doc_file(doc_path)

        # text = None
        # text = ' '.join([' '.join(sentence.representation()) for sentence in sentences])
        # print('vals', vocab_size, length, sentence_nr)
        return Document(doc_id=doc_id, text="", title=title, language=Language.get_from_str(language),
                        authors=authors, date=date, genres=genres, sentences=sentences, file_path=doc_path,
                        length=length, vocab_size=vocab_size, sentence_nr=sentence_nr)

    def get_document_entities_representation(self, lemma=False, lower=False, as_id=False):
        if as_id:
            return defaultdict(lambda: [], {entity_type: [(sent_id, token_id)
                                                          for (sent_id, token_id, token) in tokens]
                                            for entity_type, tokens in self.doc_entities.items()})
        else:
            return defaultdict(lambda: [], {entity_type: [token.representation(lemma=lemma, lower=lower)
                                                          for (sent_id, token_id, token) in tokens]
                                            for entity_type, tokens in self.doc_entities.items()})

    def get_wordnet_matches(self, wordnet_input: Set[str], as_id: bool = False,
                            lemma: bool = False,
                            lower: bool = False):
        matches = []
        if as_id:
            for i, sentence in enumerate(self.sentences):
                for j, token in enumerate(sentence.tokens):
                    if token.representation(lemma=True, lower=True) in wordnet_input:
                        matches.append((i, j))

        else:
            try:
                for sentence in self.sentences:
                    for j, token in sentence.tokens:
                        if token.representation(lemma=True, lower=True) in wordnet_input:
                            matches.append(token.representation(lemma=lemma, lower=lower))
            except TypeError:
                for sentence in self.sentences:
                    for token in sentence.tokens:
                        if token.representation(lemma=True, lower=True) in wordnet_input:
                            matches.append(token.representation(lemma=lemma, lower=lower))

        return matches

    def into_chunks(self, chunk_size: int):
        def flush_chunk():
            chunk_tokens = [token for current_sentence in current_sentences for token in current_sentence.tokens]
            # print(chunk_tokens)
            if len(chunk_tokens) > chunk_size:
                assert len(current_sentences) == 1
                old_length = len(current_sentences[0])
                current_sentences[0].truncate(chunk_size)
                print(f'Warning: sentences of length {old_length} truncated too {chunk_size}'
                      f', new length is {len(current_sentences[0].tokens)}')
                # raise UserWarning(f"Chunk size of {chunk_size} is too small, "
                #                   f"sentence with {len(chunk_tokens)} encountered")
            chunked_doc_id = f'{self.doc_id}_{chunk_counter}'
            return Document(doc_id=chunked_doc_id, text="",
                            title=self.title,
                            language=self.language,
                            authors=self.authors, date=self.date, genres=self.genres,
                            sentences=current_sentences,
                            file_path=self.file_path, length=len(chunk_tokens),
                            vocab_size=len(set(chunk_tokens)), sentence_nr=len(current_sentences)), len(chunk_tokens)

        current_size = 0
        current_sentences = []
        chunk_counter = 0
        self.load_sentences_from_disk()
        counted_tokens = []
        chunk_token_nrs = []
        for sentence in self.sentences:
            counted_tokens.append(len(sentence.tokens))
            if current_size + len(sentence.tokens) <= chunk_size:
                # print('o',  len(sentence.tokens), current_size)
                current_sentences.append(sentence)
                current_size += len(sentence.tokens)
            else:
                # print('x', len(sentence.tokens), current_size)
                chunk_doc, chunk_token_nr = flush_chunk()
                chunk_token_nrs.append(chunk_token_nr)
                yield chunk_doc

                current_sentences = [sentence]
                current_size = len(sentence.tokens)
                chunk_counter += 1

        if len(current_sentences) > 0:
            chunk_doc, chunk_token_nr = flush_chunk()
            chunk_token_nrs.append(chunk_token_nr)
            yield chunk_doc
        assert sum(counted_tokens) == sum(chunk_token_nrs)
        self.sentences = []


class Corpus:
    __slots__ = 'name', 'language', 'document_entities', 'series_dict', 'root_corpus_path', 'corpus_path', \
                'shared_attributes_dict', \
                'reversed_attributes_dict', 'success_dict', 'documents', 'file_dict'

    def __init__(self, source: Union[Dict[Union[str, int], Document], List[Document], str],
                 name: str = None,
                 language: Language = None):
        self.name = name
        self.language = language
        self.document_entities = None
        self.series_dict = None
        self.root_corpus_path = None
        self.shared_attributes_dict = None
        self.reversed_attributes_dict = None
        self.success_dict = None,
        self.file_dict = None
        self.corpus_path = None

        # self.corpus_storage_path = None
        if isinstance(source, str):
            # documents = self.load_corpus_documents(path=source)
            logging.info(f'try to load serialized corpus file {source}')
            if source.endswith('.json'):
                documents, name, language, document_entities, series_dict = self.load_corpus(path=source)
                self.name = name
                self.language = language
                self.document_entities = document_entities
                self.series_dict = series_dict
            else:
                other_corpus = self.fast_load(path=source)
                self.name = other_corpus.name
                self.language = other_corpus.language
                self.document_entities = other_corpus.document_entities
                self.series_dict = other_corpus.series_dict
                self.corpus_path = source
                documents = other_corpus.documents
        else:
            if name is None or language is None:
                raise UserWarning("No name or language set!")
            documents = source

        if isinstance(documents, dict):
            self.documents: Dict[str, Document] = documents
        elif isinstance(documents, list):
            self.documents: Dict[str, Document] = {document.doc_id: document for document in documents}
        else:
            self.documents: Dict[str, Document] = {}
            raise NotImplementedError("Not supported Document collection!")

        self.file_dict = {doc_id: document.file_path for doc_id, document in self.documents.items()}
        # self.token_number = sum((len(doc) for doc in self.documents.values()))

    def get_documents(self, as_list=True) -> Union[List[Document], Dict[Union[str, int], Document]]:
        if as_list:
            return list(self.documents.values())
        else:
            return self.documents

    def get_n_documents_as_corpus(self, n: int) -> "Corpus":
        documents = self.get_documents(as_list=True)
        documents = documents[:n]
        return Corpus(source=documents,
                      language=self.language,
                      name=f'{self.name}_top{n}')

    def save_corpus(self, path: str):
        document_data = {doc_id: doc.__dict__ for doc_id, doc in self.documents.items()}
        data = {"name": self.name, "language": self.language,
                "documents": document_data,
                "document_entities": self.document_entities,
                "series_dict": self.series_dict}
        # data = {doc.doc_id: doc.__dict__ for doc in self.get_documents()}

        with open(f'{path}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1, default=lambda o: o.__dict__)
        logging.info(f'saved {path}')

    def save_corpus_meta(self, corpus_dir):
        if self.root_corpus_path is None:
            self.root_corpus_path = corpus_dir
        data = {"name": self.name, "root_corpus_path": self.root_corpus_path,
                "language": self.language, "series_dict": self.series_dict,
                "success_dict": self.success_dict}
        with open(os.path.join(corpus_dir, "meta_info.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1, default=lambda o: o.__dict__)

    def save_corpus_adv(self, corpus_dir: str):
        if not os.path.isdir(corpus_dir):
            os.mkdir(corpus_dir)
        for doc_id, document in tqdm(self.documents.items(), total=len(self.documents), desc="save corpus",
                                     disable=False):
            document.store_to_corpus_file(corpus_dir)

        self.save_corpus_meta(corpus_dir)

    @staticmethod
    def load_corpus_from_dir_format(corpus_dir: str):
        # print(corpus_dir)
        meta_path = os.path.join(corpus_dir, "meta_info.json")
        with open(meta_path, 'r', encoding='utf-8') as file:
            meta_data = json.loads(file.read())

        document_paths = [file_path for file_path in os.listdir(corpus_dir) if file_path.endswith('.txt')]

        documents = [Document.create_document_from_doc_file(os.path.join(corpus_dir, doc_path), disable_sentences=True)
                     for doc_path in tqdm(document_paths, desc="load_file", disable=False)]

        corpus = Corpus(source=documents, name=meta_data["name"], language=meta_data["language"])

        corpus.root_corpus_path = meta_data["root_corpus_path"]
        # print(corpus.root_corpus_path)
        corpus.set_series_dict(meta_data["series_dict"])
        if "success_dict" in meta_data.keys():
            corpus.success_dict = meta_data["success_dict"]

        corpus.corpus_path = corpus_dir

        return corpus

    @staticmethod
    def fast_load(number_of_subparts=None, size=None, data_set=None, filer_mode=None, fake_real=None, path=None,
                  load_entities: bool = True):
        if path is None:
            corpus_dir = Corpus.build_corpus_dir(number_of_subparts,
                                                 size,
                                                 data_set,
                                                 filer_mode,
                                                 fake_real)
            if os.path.exists(corpus_dir):
                corpus = Corpus.load_corpus_from_dir_format(corpus_dir)
            else:
                corpus_path = Corpus.build_corpus_file_name(number_of_subparts,
                                                            size,
                                                            data_set,
                                                            filer_mode,
                                                            fake_real)
                corpus = Corpus(corpus_path)
                corpus.save_corpus_adv(corpus_dir)
            corpus.corpus_path = corpus_dir
        else:
            if os.path.exists(path):
                corpus = Corpus.load_corpus_from_dir_format(path)
                corpus.corpus_path = path
            elif os.path.exists(f'{path}.json'):
                corpus = Corpus(f'{path}.json')
                corpus.save_corpus_adv(path)
            else:
                raise FileNotFoundError
        # corpus.set_sentences_from_own_gens()
        if load_entities:
            corpus.set_document_entities()

        return corpus

    def get_years(self) -> [str]:
        years = set()
        for d in self.get_documents(as_list=True):
            if d.date:
                years.add(d.date)
        return sorted(list(years))

    @staticmethod
    def build_corpus_file_name(number_of_subparts: Union[int, str], size: Union[int, str],
                               dataset: str, filter_mode: str, fake_series: str) -> str:
        sub_path = DataHandler.build_config_str(number_of_subparts, size, dataset, filter_mode,
                                                '', fake_series)
        return os.path.join(config["system_storage"]["corpora"], f'{sub_path}.json')

    @staticmethod
    def build_corpus_name(number_of_subparts: Union[int, str], size: Union[int, str],
                          dataset: str, filter_mode: str, fake_series: str) -> str:
        return DataHandler.build_config_str(number_of_subparts, size, dataset, filter_mode,
                                            '', fake_series)

    @staticmethod
    def build_corpus_dir(number_of_subparts: Union[int, str], size: Union[int, str],
                         dataset: str, filter_mode: str, fake_series: str) -> str:
        sub_path = DataHandler.build_config_str(number_of_subparts, size, dataset, filter_mode,
                                                '', fake_series)
        return os.path.join(config["system_storage"]["corpora"], sub_path)

    @staticmethod
    def load_corpus_documents(path: str) -> List[Document]:
        logging.info(f"load {path}")
        with open(path, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())

        corpus = [Document(doc_id=doc["doc_id"],
                           text=doc["text"],
                           title=doc["title"],
                           language=Language.get_from_str(doc["language"]),
                           authors=doc["authors"],
                           date=doc["date"],
                           genres=doc["genres"])
                  for doc in data]
        logging.info(f"{path} loaded")

        return corpus

    @staticmethod
    def load_corpus(path: str):
        logging.info(f"load {path}")
        with open(path, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())

        # doc_sents = []
        # for doc_id, doc in data["documents"].items():
        #     sentences = [Sentence([Token(text=token["text"],
        #                         lemma=token["lemma"],
        #                         pos=token["pos"],
        #                         ne=token["ne"],
        #                         punctuation=token["punctuation"],
        #                         alpha=token["alpha"],
        #                         stop=token["stop"])
        #                            for token in sentence["tokens"]])
        #                  for sentence in doc["sentences"]]
        #     doc_sents.append(sentences)
        # print(doc_sents)

        documents = {doc_id: Document(doc_id=doc["doc_id"],
                                      text=doc["text"],
                                      title=doc["title"],
                                      language=Language.get_from_str(doc["language"]),
                                      authors=doc["authors"],
                                      date=doc["date"],
                                      genres=doc["genres"],
                                      sentences=[Sentence([Token(text=token["text"],
                                                                 lemma=token["lemma"],
                                                                 pos=token["pos"],
                                                                 ne=token["ne"],
                                                                 punctuation=token["punctuation"],
                                                                 alpha=token["alpha"],
                                                                 stop=token["stop"])
                                                           for token in sentence["tokens"]])
                                                 for sentence in doc["sentences"]])
                     for doc_id, doc in data["documents"].items()}
        language = data["language"]
        name = data["name"]

        document_entities = {doc_id: defaultdict(list, {en: [Token(text=token["text"],
                                                                   lemma=token["lemma"],
                                                                   pos=token["pos"],
                                                                   ne=token["ne"],
                                                                   punctuation=token["punctuation"],
                                                                   alpha=token["alpha"],
                                                                   stop=token["stop"]) for token in tokens]
                                                        for en, tokens in doc_data.items()})
                             for doc_id, doc_data in data["document_entities"].items()}

        if data["series_dict"] is None:
            series_dict = defaultdict(list)
        else:
            series_dict = defaultdict(list, data["series_dict"])

        logging.info(f"{path} loaded")

        return documents, name, language, document_entities, series_dict

    def get_corpus_vocab(self, from_disk=True, lemma: bool = False, lower: bool = False,
                         lda_mode: bool = False):
        vocab = set()

        for document in tqdm(self.documents.values(),
                             total=len(self.documents),
                             desc="Load vocab",
                             disable=True):
            vocab.update(document.get_vocab(from_disk, lemma=lemma, lower=lower, lda_mode=lda_mode))

        if 'del' in vocab:
            vocab.remove('del')
        return vocab

    def get_index_dict(self):
        return {i: word for i, word in enumerate(self.get_corpus_vocab())}

    # def year_wise(self, ids: bool = False) -> Dict[int, List[Union[str, int, Document]]]:
    #     year_bins = defaultdict(list)
    #
    #     for doc in self.get_documents():
    #         if ids:
    #             year_bins[doc.date].append(doc.doc_id)
    #         else:
    #             year_bins[doc.date].append(doc)
    #
    #     return year_bins

    def sample(self, number_documents=100, seed=None):
        if len(self.documents) < number_documents:
            return self

        if seed:
            random.seed(seed)

        result = Corpus(source=random.sample(self.get_documents(), k=number_documents),
                        language=self.language,
                        name=f'{self.name}_{number_documents}_sample')

        return result

    def get_texts_and_doc_ids(self, texts_from_file: bool = False):
        if texts_from_file:
            return map(list, zip(*((document.get_text_from_disk(), doc_id)
                                   for doc_id, document in self.documents.items())))
        else:
            return map(list, zip(*[(document.text, doc_id) for doc_id, document in self.documents.items()]))

    def id2desc(self, index: Union[str, int]):
        if index.endswith('_sum'):
            index = index.replace('_sum', '')
        elif index.endswith('_time'):
            index = index.replace('_time', '')
        elif index.endswith('_loc'):
            index = index.replace('_loc', '')
        elif index.endswith('_atm'):
            index = index.replace('_atm', '')
        elif index.endswith('_sty'):
            index = index.replace('_sty', '')
        elif index.endswith('_plot'):
            index = index.replace('_plot', '')
        elif index.endswith('_cont'):
            index = index.replace('_cont', '')
        elif index.endswith('_raw'):
            index = index.replace('_raw', '')
        return str(self.documents[index])

    def give_spacy_lan_model(self):
        if self.language == Language.EN:
            logging.info(f"Language {Language.EN} detected.")
            return spacy.load("en_core_web_sm")
        else:
            logging.info(f"Language {Language.DE} detected.")
            return spacy.load("de_core_news_sm")

    def set_document_entities(self):
        # ents = {e.text: e.label_ for e in doc.ents}
        # entities_of_documents.append(ents)
        entities_dict = {}
        for doc_id, doc in self.documents.items():
            if doc.doc_entities is None:
                raise UserWarning("No doc entities set!")
            entities_dict[doc_id] = doc.doc_entities
        # print(entities_dict)
        self.document_entities = entities_dict

    def update_time_entities(self, update_dict: Dict[str, List[str]]):
        def find_sub_list(sub_list, main_list):
            results = []
            sll = len(sub_list)
            for ind in (j for j, e in enumerate(main_list) if e == sub_list[0]):
                if main_list[ind:ind + sll] == sub_list:
                    results.append((ind, ind + sll - 1))

            return results

        for doc_id, time_ents in tqdm(update_dict.items(), total=len(update_dict)):
            time_entities = set(time_ents)
            if doc_id in self.documents.keys():
                self.documents[doc_id].load_sentences_from_disk()
                token_reprs = [token.representation() for sentence in self.documents[doc_id].sentences
                               for token in sentence.tokens]
                for time_entity in time_entities:
                    tm = time_entity.split(' ')
                    positions = find_sub_list(tm, token_reprs)
                    for position in positions:
                        start, end = position
                        for i in range(start, end + 1):
                            self.documents[doc_id].get_token_at_doc_position(i).ne = "TIME"

                self.documents[doc_id].store_to_corpus_file(self.corpus_path)
                self.documents[doc_id].sentences = None

    def set_series_dict(self, series_dict: Dict[str, List[str]]):
        self.series_dict = series_dict

    def get_document_entities_representation(self, lemma=False, lower=False):
        return {doc_id: {entity_type: [token.representation(lemma=lemma, lower=lower) for token in tokens]
                         for entity_type, tokens in entities.items()}
                for doc_id, entities in self.document_entities.items()}

    # def set_document_entities(self, entities_dict: Dict[str, List[str]]):
    #     self.document_entities = entities_dict

    def set_sentences(self, sentences: Dict[str, List[Sentence]]):
        # for doc_id, document in self.documents.items():
        #     document.set_sentences(sentences[doc_id])
        [document.set_sentences(sentences[doc_id]) for doc_id, document in self.documents.items()]

    # def set_sentences_from_own_gens(self):
    #     # for doc_id, document in self.documents.items():
    #     #     document.set_sentences(sentences[doc_id])
    #     [document.set_sentences_from_gen() for document in tqdm(self.documents.values(), desc="exhaust gen")]

    # def set_root_path(self, root_path: str):
    #     self.root_corpus_path = root_path
    def get_flat_documents(self, lemma: bool = False, lower: bool = False, as_sentence: bool = True):
        if as_sentence:
            documents = [' '.join(document.get_flat_document_tokens(lemma, lower))
                         for doc_id, document in self.documents.items()]
        else:
            documents = [document.get_flat_document_tokens(lemma, lower) for doc_id, document in self.documents.items()]
        if len(documents) == 0:
            raise UserWarning("No sentences set")

        return documents

    def get_flat_document_tokens(self, lemma: bool = False, lower: bool = False, as_dict: bool = False,
                                 generator: bool = False) -> Union[List[str], Dict[str, List[str]]]:
        # for doc_id, document in self.documents.items():
        #     print(document.sentences[1].tokens)
        if not generator:
            if as_dict:
                tokens = {doc_id: [token.representation(lemma, lower)
                                   for sentence in document.sentences for token in sentence.tokens]
                          for doc_id, document in self.documents.items()}
            else:
                tokens = [[token.representation(lemma, lower) for sentence in document.sentences
                           for token in sentence.tokens]
                          for doc_id, document in self.documents.items()]
            if len(tokens) == 0:
                raise UserWarning("No sentences set")
        else:
            if as_dict:
                tokens = {doc_id: (token.representation(lemma, lower)
                                   for sentence in document.get_sentences_from_disk() for token in sentence.tokens)
                          for doc_id, document in self.documents.items()}
            else:
                tokens = [(token.representation(lemma, lower) for sentence in document.get_sentences_from_disk()
                           for token in sentence.tokens)
                          for doc_id, document in self.documents.items()]
        return tokens

    def get_tokens_from_file(self, doc_id):
        return Document.create_document_from_doc_file(self.file_dict[doc_id])

    def get_improved_flat_document_tokens(self, lemma: bool = False, lower: bool = False, as_dict: bool = False) -> \
            Union[List[str], Dict[str, List[str]]]:
        if as_dict:
            tokens = {doc_id: (token.representation(lemma, lower)
                               for sentence in self.get_tokens_from_file(doc_id).sentences for token in sentence.tokens)
                      for doc_id in tqdm(self.documents.keys(), total=len(self.documents), desc="Get tokens of doc",
                                         disable=True)}
        else:
            tokens = ((token.representation(lemma, lower)
                       for sentence in self.get_tokens_from_file(doc_id).sentences for token in sentence.tokens)
                      for doc_id in self.documents.keys())
        if len(tokens) == 0:
            raise UserWarning("No sentences set")
        return tokens

    def get_flat_and_filtered_document_tokens(self, lemma: bool = False, lower: bool = False, pos: list = None,
                                              focus_stopwords: bool = False,
                                              focus_punctuation: bool = False,
                                              focus_ne: bool = False,
                                              masking: bool = False,
                                              revert: bool = False):
        def filter_condition(token: Token):
            if revert:
                return (not focus_stopwords or not token.stop) \
                       and (not focus_punctuation or not token.alpha) \
                       and (not pos or token.pos not in pos) \
                       and (not focus_ne or not token.ne)
            else:
                return (not focus_stopwords or token.stop) \
                       and (not focus_punctuation or token.alpha) \
                       and (not pos or token.pos in pos) \
                       and (not focus_ne or token.ne)

        def mask(input_token: Token):
            output_token = Token(text=input_token.text,
                                 lemma=input_token.lemma,
                                 pos=input_token.pos,
                                 ne=input_token.ne,
                                 punctuation=input_token.punctuation,
                                 alpha=input_token.alpha,
                                 stop=input_token.stop)
            if not filter_condition(output_token):
                output_token.text = "del"
                output_token.lemma = "del"
            return output_token

        # for doc_id, document in self.documents.items():
        #     print(document.sentences[1].tokens)
        if not masking:
            tokens = [[token.representation(lemma, lower)
                       for sentence in document.sentences
                       for token in sentence.tokens
                       if filter_condition(token)]
                      for doc_id, document in self.documents.items()]
        else:
            tokens = [[mask(token).representation(lemma, lower)
                       for sentence in document.sentences
                       for token in sentence.tokens]
                      for doc_id, document in self.documents.items()]

        tokens = [token for doc in tokens for token in doc if len(doc) > 0]

        if len(tokens) == 0:
            raise UserWarning("No sentences set")

        return tokens

    def get_flat_and_filtered_document_tokens_from_disk(self, lemma: bool = False, lower: bool = False,
                                                        pos: list = None,
                                                        focus_stopwords: bool = False,
                                                        focus_punctuation: bool = False,
                                                        focus_ne: bool = False,
                                                        masking: bool = False,
                                                        revert: bool = False):
        def filter_condition(token: Token):
            if revert:
                return (not focus_stopwords or not token.stop) \
                       and (not focus_punctuation or not token.alpha) \
                       and (not pos or token.pos not in pos) \
                       and (not focus_ne or not token.ne)
            else:
                return (not focus_stopwords or token.stop) \
                       and (not focus_punctuation or token.alpha) \
                       and (not pos or token.pos in pos) \
                       and (not focus_ne or token.ne)

        def mask(input_token: Token):
            output_token = Token(text=input_token.text,
                                 lemma=input_token.lemma,
                                 pos=input_token.pos,
                                 ne=input_token.ne,
                                 punctuation=input_token.punctuation,
                                 alpha=input_token.alpha,
                                 stop=input_token.stop)
            if not filter_condition(output_token):
                output_token.text = "del"
                output_token.lemma = "del"
            return output_token

        # for doc_id, document in self.documents.items():
        #     print(document.sentences[1].tokens)
        if not masking:
            tokens = [[token.representation(lemma, lower)
                       for sentence in document.get_sentences_from_disk()
                       for token in sentence.tokens
                       if filter_condition(token)]
                      for doc_id, document in self.documents.items()]
        else:
            tokens = [[mask(token).representation(lemma, lower)
                       for sentence in document.get_sentences_from_disk()
                       for token in sentence.tokens]
                      for doc_id, document in self.documents.items()]

        if len(tokens) == 0:
            raise UserWarning("No sentences set")

        return tokens

    def get_flat_and_random_document_tokens(self, prop_to_keep: float, seed: int,
                                            lemma: bool = False, lower: bool = False,
                                            masking: bool = False):
        def filter_condition():
            random_number = random.randint(1, 1000)
            return random_number <= 1000 * prop_to_keep

        def mask(input_token: Token):
            output_token = Token(text=input_token.text,
                                 lemma=input_token.lemma,
                                 pos=input_token.pos,
                                 ne=input_token.ne,
                                 punctuation=input_token.punctuation,
                                 alpha=input_token.alpha,
                                 stop=input_token.stop)
            if not filter_condition():
                output_token.text = "del"
                output_token.lemma = "del"
            return output_token

        random.seed(seed)
        if not masking:
            tokens = [[token.representation(lemma, lower)
                       for sentence in document.sentences
                       for token in sentence.tokens
                       if filter_condition()]
                      for doc_id, document in self.documents.items()]
        else:
            tokens = [[mask(token).representation(lemma, lower)
                       for sentence in document.sentences
                       for token in sentence.tokens]
                      for doc_id, document in self.documents.items()]

        if len(tokens) == 0:
            raise UserWarning("No sentences set")

        return tokens

    def get_flat_corpus_sentences(self, lemma: bool = False, lower: bool = False, generator: bool = True):
        if not generator:
            sentences = [sentence.representation(lemma, lower)
                         for doc_id, document in self.documents.items()
                         for sentence in document.sentences]
            if len(sentences) == 0:
                raise UserWarning("No sentences set")
        else:
            sentences = (sentence.representation(lemma, lower)
                         for doc_id, document in self.documents.items()
                         for sentence in document.get_sentences_from_disk())
        return sentences

    def calculate_documents_with_shared_attributes(self):
        same_author_dict = defaultdict(set)
        same_year_dict = defaultdict(set)
        same_genre_dict = defaultdict(set)

        for doc_id, document in self.documents.items():
            same_author_dict[document.authors].add(doc_id)
            same_year_dict[document.date].add(doc_id)
            same_genre_dict[document.genres].add(doc_id)
        self.shared_attributes_dict = {"same_author": same_author_dict, "same_year": same_year_dict,
                                       "same_genres": same_genre_dict}

        self.reversed_attributes_dict = {category: Utils.revert_dictionaried_set(category_dict)
                                         for category, category_dict in self.shared_attributes_dict.items()}
        # print(self.shared_attributes_dict["same_author"])

    def get_other_doc_ids_by_same_author(self, doc_id):
        # same_author_docs = [document for document in self.documents.values()
        #                     if document.doc_id != doc_id and document.authors == self.documents[doc_id].authors]
        # return same_author_docs
        if self.shared_attributes_dict is None:
            self.calculate_documents_with_shared_attributes()
        other_ids = list(self.shared_attributes_dict["same_author"][self.documents[doc_id].authors])
        if doc_id in other_ids:
            other_ids.remove(doc_id)
        return other_ids

    def get_other_doc_ids_by_same_genres(self, doc_id):
        if self.shared_attributes_dict is None:
            self.calculate_documents_with_shared_attributes()
        # print(self.shared_attributes_dict["same_genres"])
        # print(self.documents[doc_id].genres)
        other_ids = list(self.shared_attributes_dict["same_genres"][self.documents[doc_id].genres])
        # print(other_ids)
        if doc_id in other_ids:
            other_ids.remove(doc_id)
        # print(other_ids)
        return other_ids

    def get_other_doc_ids_by_same_year(self, doc_id):
        if self.shared_attributes_dict is None:
            self.calculate_documents_with_shared_attributes()
        other_ids = list(self.shared_attributes_dict["same_year"][self.documents[doc_id].date])
        if doc_id in other_ids:
            other_ids.remove(doc_id)
        return other_ids

    def get_windowed_aspects(self, aspect_dict: Dict[str, List[str]], window_size: int = 10):
        context_aspect_dict = {}
        for aspect, aspect_docs in aspect_dict.items():
            windowed_docs = []
            for (doc_id, document), aspect_doc in zip(self.documents.items(), aspect_docs):
                windowed_sentence = []

                for i, sentence in enumerate(document.sentences):
                    token_id_matches = []

                    for j, token in enumerate(sentence.tokens):
                        if token.representation() == aspect_doc[i][j]:
                            token_id_matches.append(j)
                    for matched_id in token_id_matches:
                        min_id = matched_id - window_size
                        max_id = matched_id + window_size
                        if min_id < 0:
                            min_id = 0
                        if max_id > len(sentence.tokens) - 1:
                            max_id = len(sentence.tokens) - 1

                        windowed_sentence.extend(sentence.tokens[min_id, max_id])
                windowed_docs.append(windowed_sentence)

            context_aspect_dict[aspect] = windowed_docs
        return context_aspect_dict

    def fake_series(self, series_corpus_dir: str, number_of_sub_parts: int) -> Tuple["Corpus", Dict[str, List[str]]]:
        fake_series_corpus = []
        fake_series_dict = defaultdict(list)

        file_dict = {}
        for doc_id, document in self.documents.items():
            # doc_sentences = document.sentences
            doc_sentences = document.get_sentences_from_disk()
            # print(len(doc_sentences), number_of_sub_parts)
            if len(doc_sentences) < number_of_sub_parts:
                raise UserWarning("Nr of document sentences too small!")
            sentence_counter = 0
            # avg_doc_length = math.ceil(len(doc_sentences) / number_of_sub_parts)
            avg_doc_length = len(doc_sentences) // number_of_sub_parts
            # print(doc_id, len(doc_sentences))
            for i in range(0, number_of_sub_parts):
                series_doc_id = f'{doc_id}_{i}'
                fake_series_dict[doc_id].append(series_doc_id)
                fake_series_doc = Document(doc_id=series_doc_id,
                                           text=document.text,
                                           title=f'{document.title} {i}',
                                           language=document.language,
                                           authors=document.authors,
                                           date=document.date,
                                           genres=document.genres)
                if i + 1 == number_of_sub_parts:
                    end = None
                else:
                    end = (i + 1) * avg_doc_length
                sub_sentences = doc_sentences[i * avg_doc_length:end]
                fake_series_doc.set_sentences(sub_sentences)
                fake_series_doc.calculate_sizes()
                fake_series_doc.file_path = os.path.join(series_corpus_dir,
                                                         f'{fake_series_doc.meta_string_representation()}.txt')
                file_dict[doc_id] = fake_series_doc.file_path
                # new_document.set_entities()
                fake_series_doc.store_to_corpus_file(series_corpus_dir)
                fake_series_doc.sentences = None
                del sub_sentences

                # fake_series_doc.reset_text_based_on_sentences()
                # if len(fake_series_doc.sentences) == 0:
                #     print(document.date, document.doc_id, document.title, fake_series_doc.text, document.text)
                #     print(sentence_counter, len(doc_sentences), avg_doc_length)
                fake_series_corpus.append(fake_series_doc)
                sentence_counter += fake_series_doc.sentences_nr

                assert fake_series_doc.sentences_nr > 0
            assert sentence_counter == len(doc_sentences)

        # fake_series_corpus.set_document_entities()
        fake_series_corpus = Corpus(fake_series_corpus, name=f'{self.name}_fake', language=self.language)
        # fake_series_corpus.set_document_entities()
        fake_series_corpus.file_dict = file_dict
        fake_series_corpus.set_series_dict(fake_series_dict)
        fake_series_corpus.save_corpus_meta(series_corpus_dir)
        # for doc_id, doc in fake_series_corpus.documents.items():
        #     print(doc_id, doc.text)
        # print(fake_series_corpus)
        return fake_series_corpus, fake_series_dict

    def get_common_words_relaxed(self, series_dictionary: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        common_words = CommonWords.relaxed(series_dictionary, self.get_flat_document_tokens(as_dict=True))
        for series_id, doc_ids in series_dictionary.items():
            for doc_id in doc_ids:
                common_words[doc_id] = common_words[series_id]
        return common_words

    def get_common_words_strict(self, series_dictionary: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        common_words = CommonWords.strict(series_dictionary, self.get_flat_document_tokens(as_dict=True))
        for series_id, doc_ids in series_dictionary.items():
            for doc_id in doc_ids:
                common_words[doc_id] = common_words[series_id]
        return common_words

    def get_common_words_relaxed_gen_words(self, series_dictionary: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        common_words = CommonWords.relaxed_general_words_sensitive(series_dictionary,
                                                                   self.get_flat_document_tokens(as_dict=True))
        for series_id, doc_ids in series_dictionary.items():
            for doc_id in doc_ids:
                common_words[doc_id] = common_words[series_id]
        return common_words

    def get_common_words_strict_gen_words(self, series_dictionary: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        common_words = CommonWords.strict_general_words_sensitive(series_dictionary,
                                                                  self.get_flat_document_tokens(as_dict=True))
        for series_id, doc_ids in series_dictionary.items():
            for doc_id in doc_ids:
                common_words[doc_id] = common_words[series_id]
        return common_words

    def moderate_specific_word_reduction(self) -> Set[str]:
        author_dict = defaultdict(list)
        for doc_id, document in self.documents.items():
            author_dict[document.authors].append(doc_id)
        absolute_threshold = math.ceil(np.max([len(doc_ids) for author, doc_ids in author_dict.items() if author is not None]))
        # percentage_share = 0.00186
        # absolute_threshold = math.ceil(len(self.documents) * percentage_share)
        if absolute_threshold < 2:
            absolute_threshold = 2
        specific_words = CommonWords.global_too_specific_words_doc_frequency(self,
                                                                             absolute_share=absolute_threshold)
        return specific_words

    def strict_specific_word_reduction(self) -> Set[str]:
        author_dict = defaultdict(list)
        for doc_id, document in self.documents.items():
            author_dict[document.authors].append(doc_id)
        absolute_threshold = np.max([len(doc_ids) for author, doc_ids in author_dict.items() if author is not None])

        # percentage_share = 0.02423
        # absolute_threshold = math.ceil(len(self.documents) * percentage_share)
        if absolute_threshold < 3:
            absolute_threshold = 3
        specific_words = CommonWords.global_too_specific_words_doc_frequency(self,
                                                                             absolute_share=absolute_threshold)
        return specific_words

    # def get_common_words(self, series_dictionary: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    #     common_words = defaultdict(set)
    #     for series_id, doc_ids in series_dictionary.items():
    #         series_words = []
    #         for doc_id in doc_ids:
    #             series_words.append(set(self.documents[doc_id].get_flat_document_tokens(lemma=True, lower=True)))
    #
    #         for token_set_a in series_words:
    #             for token_set_b in series_words:
    #                 if token_set_a != token_set_b:
    #                     common_words[series_id].update(token_set_a.intersection(token_set_b))
    #         # common_words[series_id] = set.intersection(*series_words)
    #         for doc_id in doc_ids:
    #             common_words[doc_id] = common_words[series_id]
    #     return common_words

    def common_words_corpus_filtered(self, common_words: Union[Set[str], Dict[str, Set[str]]], masking: bool):
        def filter_condition(token: Token, document_id: str, common_ws: Union[Set[str], Dict[str, Set[str]]]):
            # print(token.representation(lemma=True, lower=True) not in common_words[document_id],
            #       token.representation(lemma=True, lower=True),
            #       common_words[document_id])
            if isinstance(common_words, set):
                return token.representation(lemma=False, lower=False) not in common_ws
            else:
                return token.representation(lemma=False, lower=False) not in common_ws[document_id]

        def mask(token: Token, document_id: str, common_ws: Set[str]):
            if not filter_condition(token, document_id, common_ws):
                token.text = "del"
                token.lemma = "del"
            return token

        for doc_id, document in tqdm(self.documents.items(), total=len(self.documents), desc="Filtering corpus",
                                     disable=True):
            if not masking:
                new_sents_gen = (Sentence([token for token in sentence.tokens
                                           if filter_condition(token, doc_id, common_words)])
                                 for sentence in document.sentences)
                new_sents = []
                for sent in new_sents_gen:
                    if len(sent.tokens) == 0:
                        sent.tokens.append(Token.empty_token())
                    new_sents.append(sent)
                # for new_s in new_sents:
                #     print(new_s.representation())
            else:
                new_sents = [Sentence([mask(token, doc_id, common_words) for token in sentence.tokens])
                             for sentence in document.sentences]
                # for new_s in new_sents:
                #     print(new_s.representation())
            document.set_sentences(new_sents)

        #     documents[doc_id] = Document(doc_id=doc_id,
        #                                  text=document.text,
        #                                  title=document.title,
        #                                  language=document.language,
        #                                  authors=document.authors,
        #                                  date=document.date,
        #                                  genres=document.genres,
        #                                  sentences=new_sents)
        #
        # common_words_corpus = Corpus(documents, self.name, self.language)
        # common_words_corpus.set_series_dict(self.series_dict)
        # common_words_corpus.set_document_entities()
        # common_words_corpus.file_dict = self.file_dict

    def common_words_corpus_copy(self, common_words: Union[Set[str], Dict[str, Set[str]]], masking: bool):
        def filter_condition(token: Token, document_id: str, common_ws: Union[Set[str], Dict[str, Set[str]]]):
            # print(token.representation(lemma=True, lower=True) not in common_words[document_id],
            #       token.representation(lemma=True, lower=True),
            #       common_words[document_id])
            if isinstance(common_words, set):
                return token.representation(lemma=False, lower=False) not in common_ws
            else:
                return token.representation(lemma=False, lower=False) not in common_ws[document_id]

        def mask(token: Token, document_id: str, common_ws: Set[str]):
            if not filter_condition(token, document_id, common_ws):
                token.text = "del"
                token.lemma = "del"
            return token

        documents = {}
        for doc_id, document in tqdm(self.documents.items(), total=len(self.documents), desc="Filtering corpus",
                                     disable=True):
            if not masking:
                new_sents_gen = (Sentence([token for token in sentence.tokens
                                           if filter_condition(token, doc_id, common_words)])
                                 for sentence in document.sentences)
                new_sents = []
                for sent in new_sents_gen:
                    if len(sent.tokens) == 0:
                        sent.tokens.append(Token.empty_token())
                    new_sents.append(sent)
                # for new_s in new_sents:
                #     print(new_s.representation())
            else:
                new_sents = [Sentence([mask(token, doc_id, common_words) for token in sentence.tokens])
                             for sentence in document.sentences]
                # for new_s in new_sents:
                #     print(new_s.representation())
            # document.set_sentences(new_sents)

            documents[doc_id] = Document(doc_id=doc_id,
                                         text=document.text,
                                         title=document.title,
                                         language=document.language,
                                         authors=document.authors,
                                         date=document.date,
                                         genres=document.genres,
                                         sentences=new_sents)

        common_words_corpus = Corpus(documents, self.name, self.language)
        common_words_corpus.set_series_dict(self.series_dict)
        common_words_corpus.set_document_entities()
        common_words_corpus.file_dict = self.file_dict
        return common_words_corpus

    @staticmethod
    def swap_corpus_dir(old_path, new_dir):
        return os.path.join(new_dir, os.path.basename(old_path))

    def common_words_corpus_copy_mem_eff(self, common_words: Union[Set[str], Dict[str, Set[str]]], masking: bool,
                                         corpus_dir: str, through_no_sentences_error: bool = True):
        def filter_condition(token: Token, document_id: str, common_ws: Union[Set[str], Dict[str, Set[str]]]):
            # print(token.representation(lemma=True, lower=True) not in common_words[document_id],
            #       token.representation(lemma=True, lower=True),
            #       common_words[document_id])
            if isinstance(common_words, set):
                return token.representation(lemma=False, lower=False) not in common_ws
            else:
                return token.representation(lemma=False, lower=False) not in common_ws[document_id]

        def mask(token: Token, document_id: str, common_ws: Set[str]):
            if not filter_condition(token, document_id, common_ws):
                token.text = "del"
                token.lemma = "del"
            return token

        documents = {}
        file_dict = {}
        for doc_id, document in tqdm(self.documents.items(), total=len(self.documents), desc="Filtering corpus",
                                     disable=True):

            if not masking:
                new_sents_gen = (Sentence([token for token in sentence.tokens
                                           if filter_condition(token, doc_id, common_words)])
                                 for sentence in document.get_sentences_from_disk())
                new_sents = []
                for sent in new_sents_gen:
                    if len(sent.tokens) == 0:
                        sent.tokens.append(Token.empty_token())
                    new_sents.append(sent)
                # for new_s in new_sents:
                #     print(new_s.representation())
            else:
                new_sents = [Sentence([mask(token, doc_id, common_words) for token in sentence.tokens])
                             for sentence in document.get_sentences_from_disk()]
                # for new_s in new_sents:
                #     print(new_s.representation())
            # document.set_sentences(new_sents)

            new_document = Document(doc_id=doc_id,
                                    text="",
                                    title=document.title,
                                    language=document.language,
                                    authors=document.authors,
                                    date=document.date,
                                    genres=document.genres,
                                    sentences=new_sents, )

            new_document.calculate_sizes(through_error=through_no_sentences_error)
            # new_document.set_entities()

            new_document.file_path = os.path.join(corpus_dir, f'{new_document.meta_string_representation()}.txt')
            file_dict[doc_id] = new_document.file_path
            new_document.store_to_corpus_file(corpus_dir)

            new_document.sentences = None
            del new_sents
            documents[doc_id] = new_document

        common_words_corpus = Corpus(documents, self.name, self.language)
        common_words_corpus.set_series_dict(self.series_dict)
        # common_words_corpus.set_document_entities()

        common_words_corpus.file_dict = file_dict

        common_words_corpus.save_corpus_meta(corpus_dir)
        return common_words_corpus

    # def filter(self, mode: str, masking: bool = False, common_words: Dict[str, Set[str]] = None):
    #     def filter_condition(token: Token, document_id: str):
    #         # print(token.representation(lemma=True, lower=True) not in common_words[document_id],
    #         #       token.representation(lemma=True, lower=True),
    #         #       common_words[document_id])
    #         return token.representation(lemma=False, lower=False) not in common_words[document_id]
    #
    #     def mask(token: Token, document_id: str):
    #         if not filter_condition(token, document_id):
    #             token.text = "del"
    #             token.lemma = "del"
    #         return token
    #
    #     if mode.lower() == "no_filter" or mode.lower() == "nf":
    #         pass
    #         # return self
    #     elif mode.lower() == "common_words" or mode.lower() == "cw":
    #         # print('>>', common_words["bs_0"])
    #         for doc_id, document in tqdm(self.documents.items(), total=len(self.documents), desc="Filtering corpus",
    #                                      disable=True):
    #             if not masking:
    #                 new_sents = [Sentence([token for token in sentence.tokens if filter_condition(token, doc_id)])
    #                              for sentence in document.sentences]
    #                 for sent in new_sents:
    #                     if len(sent.tokens) == 0:
    #                         sent.tokens.append(Token.empty_token())
    #                 # for new_s in new_sents:
    #                 #     print(new_s.representation())
    #             else:
    #                 new_sents = [Sentence([mask(token, doc_id) for token in sentence.tokens])
    #                              for sentence in document.sentences]
    #                 # for new_s in new_sents:
    #                 #     print(new_s.representation())
    #             document.set_sentences(new_sents)
    #             # return self
    #     else:
    #         pos = None
    #         remove_stopwords = False
    #         remove_punctuation = False
    #         remove_ne = False
    #         revert = False
    #
    #         if mode.lower() == "named_entities" or mode.lower() == "ne" or mode.lower() == "named_entity":
    #             remove_ne = True
    #             pos = ["PROPN"]
    #         elif mode.lower() == "nouns" or mode.lower() == "n" or mode.lower() == "noun":
    #             pos = ["NOUN", "PROPN"]
    #             remove_ne = True
    #         elif mode.lower() == "verbs" or mode.lower() == "v" or mode.lower() == "verb":
    #             pos = ["VERB", "ADV"]
    #         elif mode.lower() == "adjectives" or mode.lower() == "a" or mode.lower() == "adj" \
    #                 or mode.lower() == "adjective":
    #             pos = ["ADJ"]
    #         elif mode.lower() == "avn" or mode.lower() == "anv" or mode.lower() == "nav" or mode.lower() == "nva" \
    #                 or mode.lower() == "van" or mode.lower() == "vna":
    #             remove_ne = True
    #             pos = ["NOUN", "PROPN", "ADJ", "VERB", "ADV"]
    #         elif mode.lower() == "stopwords" or mode.lower() == "stop_words" \
    #                 or mode.lower() == "stopword" or mode.lower() == "stop_word" \
    #                 or mode.lower() == "stop" or mode.lower() == "sw":
    #             remove_stopwords = True
    #         elif mode.lower() == "punctuation" or mode.lower() == "punct" \
    #                 or mode.lower() == "." or mode.lower() == "pun" \
    #                 or mode.lower() == "punc" or mode.lower() == "zeichen":
    #             remove_punctuation = True
    #         else:
    #             raise UserWarning("Not supported mode")
    #         Preprocesser.filter(self,
    #                             pos=pos,
    #                             remove_stopwords=remove_stopwords,
    #                             remove_punctuation=remove_punctuation,
    #                             remove_ne=remove_ne,
    #                             masking=masking,
    #                             revert=revert)

    def filter_on_copy(self, mode: str, masking: bool = False) -> "Corpus":
        # def filter_condition(token: Token, document_id: str, common_ws: Dict[str, Set[str]]):
        #     # print(token.representation(lemma=True, lower=True) not in common_words[document_id],
        #     #       token.representation(lemma=True, lower=True),
        #     #       common_words[document_id])
        #     return token.representation(lemma=False, lower=False) not in common_ws[document_id]

        # def mask(token: Token, document_id: str, common_ws: Dict[str, Set[str]]):
        #     if not filter_condition(token, document_id, common_ws):
        #         token.text = "del"
        #         token.lemma = "del"
        #     return token

        if mode.lower() == "no_filter" or mode.lower() == "nf":
            return self
        elif mode.lower() == "common_words_relaxed" or mode.lower() == "cw_rel":
            common_words_of_task = self.get_common_words_relaxed(self.series_dict)
            corpus = self.common_words_corpus_copy(common_words_of_task, masking)
            return corpus
        elif mode.lower() == "common_words_strict" or mode.lower() == "cw_str":
            common_words_of_task = self.get_common_words_strict(self.series_dict)
            corpus = self.common_words_corpus_copy(common_words_of_task, masking)
            return corpus
        elif mode.lower() == "common_words_relaxed_general_words_sensitive" or mode.lower() == "cw_rel_gw":
            common_words_of_task = self.get_common_words_relaxed_gen_words(self.series_dict)
            corpus = self.common_words_corpus_copy(common_words_of_task, masking)
            return corpus
        elif mode.lower() == "common_words_strict_general_words_sensitive" or mode.lower() == "cw_str_gw":
            common_words_of_task = self.get_common_words_strict_gen_words(self.series_dict)
            corpus = self.common_words_corpus_copy(common_words_of_task, masking)
            return corpus
        elif mode.lower() == "common_words_doc_freq" or mode.lower() == "cw_df":
            common_words = self.get_global_common_words()
            corpus = self.common_words_corpus_copy(common_words, masking)
            return corpus
        else:
            pos = None
            remove_stopwords = False
            remove_punctuation = False
            remove_ne = False
            revert = False

            if mode.lower() == "named_entities" or mode.lower() == "ne" or mode.lower() == "named_entity":
                remove_ne = True
                pos = ["PROPN"]
            elif mode.lower() == "nouns" or mode.lower() == "n" or mode.lower() == "noun":
                pos = ["NOUN", "PROPN"]
                remove_ne = True
            elif mode.lower() == "verbs" or mode.lower() == "v" or mode.lower() == "verb":
                pos = ["VERB", "ADV"]
            elif mode.lower() == "adjectives" or mode.lower() == "a" or mode.lower() == "adj" \
                    or mode.lower() == "adjective":
                pos = ["ADJ"]
            elif mode.lower() == "avn" or mode.lower() == "anv" or mode.lower() == "nav" or mode.lower() == "nva" \
                    or mode.lower() == "van" or mode.lower() == "vna":
                remove_ne = True
                pos = ["NOUN", "PROPN", "ADJ", "VERB", "ADV"]
            elif mode.lower() == "stopwords" or mode.lower() == "stop_words" \
                    or mode.lower() == "stopword" or mode.lower() == "stop_word" \
                    or mode.lower() == "stop" or mode.lower() == "sw":
                remove_stopwords = True
            elif mode.lower() == "punctuation" or mode.lower() == "punct" \
                    or mode.lower() == "." or mode.lower() == "pun" \
                    or mode.lower() == "punc" or mode.lower() == "zeichen":
                remove_punctuation = True
            else:
                raise UserWarning(f"Not supported mode: {mode}")
            return Preprocesser.filter_on_copy(self,
                                               pos=pos,
                                               remove_stopwords=remove_stopwords,
                                               remove_punctuation=remove_punctuation,
                                               remove_ne=remove_ne,
                                               masking=masking,
                                               revert=revert)

    def filter_on_copy_mem_eff(self, filtered_corpus_dir: str, mode: str, masking: bool = False) -> "Corpus":
        # def filter_condition(token: Token, document_id: str, common_ws: Dict[str, Set[str]]):
        #     # print(token.representation(lemma=True, lower=True) not in common_words[document_id],
        #     #       token.representation(lemma=True, lower=True),
        #     #       common_words[document_id])
        #     return token.representation(lemma=False, lower=False) not in common_ws[document_id]

        # def mask(token: Token, document_id: str, common_ws: Dict[str, Set[str]]):
        #     if not filter_condition(token, document_id, common_ws):
        #         token.text = "del"
        #         token.lemma = "del"
        #     return token
        # todo: find good way to handle identical copy,
        #  fixme copies contain del token and count it in title
        if mode.lower() == "no_filter" or mode.lower() == "nf":
            return self.common_words_corpus_copy_mem_eff(set(), masking,
                                                         corpus_dir=filtered_corpus_dir)
        elif mode.lower() == "common_words_relaxed" or mode.lower() == "cw_rel":
            common_words_of_task = self.get_common_words_relaxed(self.series_dict)
            corpus = self.common_words_corpus_copy_mem_eff(common_words_of_task, masking,
                                                           corpus_dir=filtered_corpus_dir)
            return corpus
        elif mode.lower() == "common_words_strict" or mode.lower() == "cw_str":
            common_words_of_task = self.get_common_words_strict(self.series_dict)
            corpus = self.common_words_corpus_copy_mem_eff(common_words_of_task, masking,
                                                           corpus_dir=filtered_corpus_dir)
            return corpus
        elif mode.lower() == "common_words_relaxed_general_words_sensitive" or mode.lower() == "cw_rel_gw":
            common_words_of_task = self.get_common_words_relaxed_gen_words(self.series_dict)
            corpus = self.common_words_corpus_copy_mem_eff(common_words_of_task, masking,
                                                           corpus_dir=filtered_corpus_dir)
            return corpus
        elif mode.lower() == "common_words_strict_general_words_sensitive" or mode.lower() == "cw_str_gw":
            common_words_of_task = self.get_common_words_strict_gen_words(self.series_dict)
            corpus = self.common_words_corpus_copy_mem_eff(common_words_of_task, masking,
                                                           corpus_dir=filtered_corpus_dir)
            return corpus
        # elif mode.lower() == "common_words_doc_freq" or mode.lower() == "cw_df":
        #     common_words = self.get_global_common_words()
        #     corpus = self.common_words_corpus_copy_mem_eff(common_words, masking,
        #                                                    corpus_dir=filtered_corpus_dir)
        #     return corpus
        elif mode.lower() == "specific_words_strict" or mode.lower() == "sw_str":
            specific_words = self.strict_specific_word_reduction()
            corpus = self.common_words_corpus_copy_mem_eff(specific_words, masking,
                                                           corpus_dir=filtered_corpus_dir)
            return corpus
        elif mode.lower() == "specific_words_moderate" or mode.lower() == "sw_mod":
            specific_words = self.moderate_specific_word_reduction()
            corpus = self.common_words_corpus_copy_mem_eff(specific_words, masking,
                                                           corpus_dir=filtered_corpus_dir)
            return corpus
        else:
            pos = None
            remove_stopwords = False
            remove_punctuation = False
            remove_ne = False
            revert = False

            if mode.lower() == "named_entities" or mode.lower() == "ne" or mode.lower() == "named_entity":
                remove_ne = True
                pos = ["PROPN"]
            elif mode.lower() == "nouns" or mode.lower() == "n" or mode.lower() == "noun":
                pos = ["NOUN", "PROPN"]
                remove_ne = True
            elif mode.lower() == "verbs" or mode.lower() == "v" or mode.lower() == "verb":
                pos = ["VERB", "ADV"]
            elif mode.lower() == "adjectives" or mode.lower() == "a" or mode.lower() == "adj" \
                    or mode.lower() == "adjective":
                pos = ["ADJ"]
            elif mode.lower() == "avn" or mode.lower() == "anv" or mode.lower() == "nav" or mode.lower() == "nva" \
                    or mode.lower() == "van" or mode.lower() == "vna":
                remove_ne = True
                pos = ["NOUN", "PROPN", "ADJ", "VERB", "ADV"]
            elif mode.lower() == "stopwords" or mode.lower() == "stop_words" \
                    or mode.lower() == "stopword" or mode.lower() == "stop_word" \
                    or mode.lower() == "stop" or mode.lower() == "sw":
                remove_stopwords = True
            elif mode.lower() == "punctuation" or mode.lower() == "punct" \
                    or mode.lower() == "." or mode.lower() == "pun" \
                    or mode.lower() == "punc" or mode.lower() == "zeichen":
                remove_punctuation = True
            else:
                raise UserWarning(f"Not supported mode: {mode}")
            return Preprocesser.filter_on_copy_mem_eff(self,
                                                       corpus_dir=filtered_corpus_dir,
                                                       pos=pos,
                                                       remove_stopwords=remove_stopwords,
                                                       remove_punctuation=remove_punctuation,
                                                       remove_ne=remove_ne,
                                                       masking=masking,
                                                       revert=revert)

    # todo implement from disk for all filters
    # def filter_and_copy_on_disk(self, filter_fun, filter_mode):
    #     filtered_corpus_dir = Corpus.build_corpus_dir(number_of_subparts,
    #                                                   corpus_size,
    #                                                   data_set,
    #                                                   filter_mode,
    #                                                   fake)
    #     corpus = self.filter_on_copy(mode=filter_mode)
    #     corpus.save_corpus_adv(filtered_corpus_dir)

    def __iter__(self):
        return self.documents.values().__iter__()

    def __len__(self):
        return len(self.documents)

    def token_number(self):
        return self.token_number  # sum((len(doc) for doc in self.documents.values()))

    def __str__(self):
        return f'docs={len(self)}, lan={self.language}, name={self.name}'

    def __getitem__(self, key):
        if isinstance(key, slice):
            # do your handling for a slice object:
            # print(key.start, key.stop, key.step)
            corpus = Corpus(source=list(self.documents.values())[key.start: key.stop: key.step],
                            name=f'{self.name}_{key.start}_{key.stop}_{key.step}',
                            language=self.language)
            corpus.series_dict = self.series_dict
            corpus.root_corpus_path = self.root_corpus_path
            corpus.corpus_path = self.corpus_path

            return corpus
        elif isinstance(key, int):
            # print(key)
            return list(self.documents.values())[key]
        else:
            # Do your handling for a plain index
            # print(key)
            return self.documents[key]

    __repr__ = __str__

    def json_representation(self):
        return vars(self)

    def length_sub_corpora(self):
        lengths = []
        for document in self.documents.values():
            lengths.append(document.length)

        lengths = np.array(lengths)

        q1_of_length = int(np.quantile(lengths, q=0.333333333))
        q3_of_length = int(np.quantile(lengths, q=0.666666666))

        sub_corpora_ids = defaultdict(list)
        for doc_id, document in self.documents.items():
            if document.length < q1_of_length:
                sub_corpora_ids["short"].append(doc_id)
            if q1_of_length <= document.length < q3_of_length:
                sub_corpora_ids["medium"].append(doc_id)
            if document.length >= q3_of_length:
                sub_corpora_ids["large"].append(doc_id)

        # for key, values in sub_corpora_ids.items():
        #     print(key, len(values))
        return sub_corpora_ids

    def length_sub_corpora_of_size(self, size: str):
        relevant_ids = set(self.length_sub_corpora()[size])
        sub_corpus = Corpus({doc_id: document for doc_id, document in self.documents.items() if doc_id in relevant_ids},
                            name=f'{self.name}_{size}', language=self.language)
        sub_corpus.file_dict = {doc_id: path for doc_id, path in self.file_dict.items() if doc_id in relevant_ids}
        sub_corpus.root_corpus_path = self.root_corpus_path
        sub_corpus.series_dict = self.series_dict
        # print(sub_corpus.documents)
        return sub_corpus

    def vector_doc_id_base_in_corpus(self, vector_doc_id: str):
        for corpus_doc_id in self.documents.keys():
            if vector_doc_id == corpus_doc_id or vector_doc_id.startswith(f'{corpus_doc_id}_'):
                return True
        return False

    def to_flair_data(self, text_corpus: bool = True):
        corpus_dir = f'{self.root_corpus_path}_flair_text'
        random.seed(42)

        if not os.path.isdir(corpus_dir):
            os.mkdir(corpus_dir)

        if not text_corpus:
            for key in ["train", "dev", "test"]:
                doc_path = os.path.join(corpus_dir, f'{key}.txt')
                with open(doc_path, 'w', encoding="utf-8") as writer:
                    writer.write(f'')

            for doc_id, document in tqdm(self.documents.items()):
                sentences = document.get_sentences_from_disk()
                random.shuffle(sentences)

                for sentence in sentences:

                    nr = random.randint(0, 100)
                    if nr < 50:
                        key = "train"
                    elif 50 <= nr < 80:
                        key = "dev"
                    else:
                        key = "test"
                    doc_path = os.path.join(corpus_dir, f'{key}.txt')
                    try:
                        with open(doc_path, 'a', encoding="utf-8") as writer:
                            for token in sentence.tokens:
                                writer.write(f'{token.get_save_file_representation(doc_id)}\n')
                            writer.write("\n")
                    except PermissionError:
                        with open(doc_path, 'a', encoding="utf-8") as writer:
                            for token in sentence.tokens:
                                writer.write(f'{token.get_save_file_representation(doc_id)}\n')
                            writer.write("\n")
        else:
            split_size = 10000
            for key in ["dev", "test"]:
                doc_path = os.path.join(corpus_dir, f'{key}.txt')
                with open(doc_path, 'w', encoding="utf-8") as writer:
                    writer.write(f'')

            train_sentence_counter = 0
            split_nr = 0
            for doc_id, document in tqdm(self.documents.items()):
                sentences = document.get_sentences_from_disk()
                random.shuffle(sentences)

                for sentence in sentences:

                    nr = random.randint(0, 100)
                    if nr < 70:
                        key = "train"
                    elif 70 <= nr < 90:
                        key = "valid"
                    else:
                        key = "test"

                    if key == "train":
                        if train_sentence_counter > split_size:
                            split_nr += 1
                            train_sentence_counter = 0
                        train_dir_path = os.path.join(corpus_dir, key)
                        if not os.path.exists(train_dir_path):
                            os.mkdir(train_dir_path)
                        doc_path = os.path.join(train_dir_path, f'split_{split_nr}.txt')
                        train_sentence_counter += 1
                    else:
                        doc_path = os.path.join(corpus_dir, f'{key}.txt')
                    try:
                        with open(doc_path, 'a', encoding="utf-8") as writer:
                            writer.write(f'{" ".join(sentence.representation())}\n')
                    except PermissionError:
                        with open(doc_path, 'a', encoding="utf-8") as writer:
                            writer.write(f'{" ".join(sentence.representation())}\n')





class Preprocesser:
    # @classmethod
    # def tokenize(cls, text: Union[str, List[str]]):
    #     if isinstance(text, str):
    #         return text.split()
    #     else:
    #         return [cls.tokenize(text) for text in text]

    @staticmethod
    def chunk_text(texts, chunk_size=3):
        chunked_list = []
        chunked_texts = []
        for text in texts:
            tokens = text.split()
            number_chunks = len(tokens) // chunk_size
            if number_chunks < 1:
                number_chunks = 1

            for i in range(0, number_chunks):
                if i < number_chunks - 1:
                    chunked_texts.append(' '.join(tokens[i * chunk_size: (i + 1) * chunk_size]))
                    chunked_list.append(True)
                else:

                    if len(tokens[i * chunk_size:]) > chunk_size:
                        chunked_texts.append(' '.join(tokens[i * chunk_size: (i + 1) * chunk_size]))
                        chunked_list.append(True)
                        j = i + 1
                    else:
                        j = i

                    chunked_texts.append(' '.join(tokens[j * chunk_size:]))
                    chunked_list.append(False)

            # print(len(text))
        return chunked_texts, chunked_list

    # @staticmethod
    # def merge_chunks(chunked_texts: Union[List[str], List[List[str]], List[Dict]], chunked_list: List[bool]):
    #     unchunked = []
    #     if all(isinstance(n, dict) for n in chunked_texts):
    #         dict_usage = True
    #         unchunked_object = {}
    #     else:
    #         dict_usage = False
    #         unchunked_object = []
    #
    #     for is_chunked, text in zip(chunked_list, chunked_texts):
    #         if isinstance(text, str):
    #             unchunked_object.append(text)
    #         else:
    #             if dict_usage:
    #                 unchunked_object.update(text)
    #             else:
    #                 unchunked_object.extend(text)
    #         if not is_chunked:
    #             if isinstance(text, str):
    #                 unchunked.append(' '.join(unchunked_object))
    #             elif isinstance(text, list):
    #                 unchunked.append(unchunked_object.copy())
    #             elif isinstance(text, dict):
    #                 unchunked.append(unchunked_object.copy())
    #             else:
    #                 raise UserWarning("Not supported type!")
    #
    #             unchunked_object.clear()
    #     return unchunked

    @staticmethod
    def merge_chunks(chunked_texts: Union[List[List[Sentence]], List[Generator[Sentence, Any, None]]],
                     chunked_list: List[bool]):
        unchunked = []
        unchunked_object = []

        for is_chunked, text in zip(chunked_list, chunked_texts):
            unchunked_object.extend(text)
            if not is_chunked:
                if isinstance(text, list):
                    unchunked.append(unchunked_object.copy())
                elif isinstance(text, Generator):
                    unchunked.append(unchunked_object.copy())
                else:
                    raise UserWarning("Not supported type!")
                unchunked_object.clear()
        return unchunked

    @staticmethod
    def annotate_corpus(corpus: Corpus, without_spacy: bool = False):
        texts, doc_ids = corpus.get_texts_and_doc_ids()

        lan_model = corpus.give_spacy_lan_model()
        prep_sentences = Preprocesser.annotate_tokens(texts,
                                                      doc_ids,
                                                      lan_model,
                                                      without_spacy=without_spacy
                                                      )

        preprocessed_corpus = Corpus(corpus.documents,
                                     name=f'{corpus.name}_prep',
                                     language=corpus.language)
        preprocessed_corpus.set_sentences(prep_sentences)

        if not without_spacy:
            preprocessed_corpus.set_document_entities()
        preprocessed_corpus.set_series_dict(corpus.series_dict)

        return preprocessed_corpus

    @staticmethod
    def annotate_and_save(corpus: Corpus, corpus_dir: str, without_spacy: bool = False):
        # texts, doc_ids = corpus.get_texts_and_doc_ids(texts_from_file=True)
        #
        # # for text in texts:
        # #     print(text[:10])
        #
        # lan_model = corpus.give_spacy_lan_model()
        # prep_sentences = Preprocesser.annotate_tokens(texts,
        #                                               doc_ids,
        #                                               lan_model,
        #                                               without_spacy=without_spacy
        #                                               )
        preprocessed_corpus = Corpus(corpus.documents,
                                     name=f'{corpus.name}_prep',
                                     language=corpus.language)

        file_dict = {}
        for doc_id, document_sentences in tqdm(corpus.documents.items(),
                                               desc="annotate",
                                               total=len(corpus.documents)):

            # exist_at_path = preprocessed_corpus[doc_id].check_if_doc_at_path(corpus_dir)

            preprocessed_corpus[doc_id].sentences = Preprocesser.sentenize(
                preprocessed_corpus[doc_id].get_text_from_disk(),
                without_spacy=without_spacy,
                lan_model=preprocessed_corpus.give_spacy_lan_model()
            )

            preprocessed_corpus[doc_id].set_entities()
            preprocessed_corpus[doc_id].calculate_sizes()
            doc_path = preprocessed_corpus[doc_id].store_to_corpus_file(corpus_dir)
            file_dict[doc_id] = doc_path
            preprocessed_corpus[doc_id].sentences = None

        if not without_spacy:
            preprocessed_corpus.set_document_entities()
        preprocessed_corpus.set_series_dict(corpus.series_dict)
        preprocessed_corpus.file_dict = file_dict
        preprocessed_corpus.save_corpus_meta(corpus_dir)

        return preprocessed_corpus

        # for doc_id, document in tqdm(corpus.documents.items(), total=len(corpus.documents), desc="Filtering corpus",
        #                              disable=True):
        #
        #     new_document = Document(doc_id=doc_id,
        #                             text=document.text,
        #                             title=document.title,
        #                             language=document.language,
        #                             authors=document.authors,
        #                             date=document.date,
        #                             genres=document.genres,
        #                             sentences=new_sents, )
        #
        #     new_document.calculate_sizes()
        #
        #     new_document.file_path = os.path.join(corpus_dir, f'{new_document.meta_string_representation()}.txt')
        #     file_dict[doc_id] = new_document.file_path
        #     new_document.store_to_corpus_file(corpus_dir)
        #
        #     new_document.sentences.clear()
        #     documents[doc_id] = new_document
        #
        # common_words_corpus = Corpus(documents, self.name, self.language)
        # common_words_corpus.set_series_dict(self.series_dict)
        # common_words_corpus.set_document_entities()
        #
        # common_words_corpus.file_dict = file_dict
        #
        # common_words_corpus.save_corpus_meta(corpus_dir)
        # return common_words_corpus

    @staticmethod
    def structure_string_texts(texts: List[str], lan_model, lemma: bool = False, lower: bool = False,
                               without_spacy: bool = False):
        prep_sentences = Preprocesser.annotate_tokens_list(texts, lan_model, without_spacy=without_spacy)

        comp_sentences = []
        for doc in prep_sentences:
            if len(doc) == 0:
                comp_sentences.append([])
            else:
                doc_sentences = []
                for sent in doc:
                    sent_repr = sent.representation(lemma=lemma, lower=lower)
                    doc_sentences.extend(sent_repr)
                comp_sentences.append(doc_sentences)

        return comp_sentences

    @staticmethod
    def annotate(list_or_gen):
        sentence_split_regex = re.compile(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<=[.?])")
        return ([Sentence([Token(text=token) for token in sentence.split() if token != ' '])
                 for sentence in re.split(sentence_split_regex, document_text)] for document_text in tqdm(list_or_gen))

    @staticmethod
    def annotate_tokens_list(texts, lan_model=None, without_spacy: bool = False) -> List[List[Sentence]]:
        def token_representation(token):
            return Token(text=token.text,
                         lemma=token.lemma_,
                         pos=token.pos_,
                         ne=token.ent_type_,
                         punctuation=token.is_punct,
                         alpha=token.is_alpha,
                         stop=token.is_stop)

        # def sentence_generator(sentence: str):
        #     yield Sentence([Token(text=token) for token in sentence.split() if token != ' '])
        #
        # def doc_generator(plain_texts):
        #     for document in tqdm(plain_texts, total=len(plain_texts), desc="Tokenize"):
        #         yield (sentence_generator(sentence)
        #                for sentence in re.split(sentence_split_regex, document))

        def nested_sentences_generator(plain_texts):
            # ds = []
            # for document in tqdm(plain_texts, total=len(plain_texts), desc="Tokenize"):
            #     ds.append((Sentence([Token(text=token) for token in sentence.split() if token != ' '])
            #                for sentence in re.split(sentence_split_regex, document)))
            ds = (((Sentence([Token(text=token) for token in sentence.split() if token != ' '])
                    for sentence in re.split(sentence_split_regex, document)))
                  for document in tqdm(plain_texts, desc="Tokenize no spacy"))
            return ds

        if without_spacy:
            sentence_split_regex = re.compile(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<=[.?])")
            nested_sentences = nested_sentences_generator(texts)
            print("gend")

        else:
            nlp = spacy.load("en_core_web_sm") if lan_model is None else lan_model
            # preprocessed_documents = []
            disable_list = ['parser']
            if not nlp.has_pipe('sentencizer'):
                nlp.add_pipe(nlp.create_pipe('sentencizer'))
            # entities_of_documents = []
            # nested_sentences = nested_sentences_spacy_generator(nlp, texts, disable_list)
            chunked_texts, chunk_list = Preprocesser.chunk_text(texts, 5000)
            chunked_texts_gen = (chunk_text for chunk_text in chunked_texts)
            logging.info(f'Start annotation of {len(chunked_texts)} chunked texts')

            nested_sentences = [[Sentence([token_representation(token)
                                           for token in sent if token.text != ' '])
                                 for sent in doc.sents]
                                for doc in tqdm(nlp.pipe(chunked_texts_gen, batch_size=100,
                                                         disable=disable_list),
                                                desc="spacify",
                                                total=len(chunked_texts))]
            nested_sentences = Preprocesser.merge_chunks(nested_sentences, chunk_list)

            # for doc in tqdm(nlp.pipe(chunked_texts_gen, disable=disable_list),
            #                 desc="spcify",
            #                 total=len(chunked_texts)):
            #     preprocessed_sentences = []
            #     for sent in doc.sents:
            #         sentence_tokens = Sentence([token_representation(token)
            #                                     for token in sent if token.text != ' '])
            #         preprocessed_sentences.append(sentence_tokens)
            #
            #     nested_sentences.append(preprocessed_sentences)
            #
            # nested_sentences = Preprocesser.merge_chunks(nested_sentences, chunk_list)

        return nested_sentences

    @staticmethod
    def annotate_tokens(texts, doc_ids, lan_model=None, without_spacy: bool = False):
        # nested_sentences = Preprocesser.annotate_tokens_list(texts, lan_model, without_spacy=without_spacy)
        nested_sentences = Preprocesser.annotate(texts)

        # for sent in nested_sentences:
        #     print(sent)
        nested_sentences_dict = {doc_id: doc_sents for doc_id, doc_sents in zip(doc_ids, nested_sentences)}
        # entities_of_documents_dict = {doc_id: doc_ents for doc_id, doc_ents in zip(doc_ids, entities_of_documents)}
        # print(nested_sentences_dict)
        # print(entities_of_documents_dict)
        # for doc_id, d in nested_sentences_dict.items():
        #     print(doc_id, d[0])
        print('dict')
        return nested_sentences_dict

    @staticmethod
    def sentenize(input_document_str: str, without_spacy: bool = True, lan_model=None):
        def token_spacy_representation(token):
            return Token(text=token.text,
                         lemma=token.lemma_,
                         pos=token.pos_,
                         ne=token.ent_type_,
                         punctuation=token.is_punct,
                         alpha=token.is_alpha,
                         stop=token.is_stop)

        # print(input_document_str)
        if without_spacy:
            sentence_split_regex = re.compile(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<=[.?])")

            # print(re.split(sentence_split_regex, input_document_str))
            # for sentence in re.split(sentence_split_regex, input_document_str):
            #     print('|', sentence, '|')
            #     for token in sentence.split():
            #         if token != ' ':
            #             pass
            #             # print(token)
            return [Sentence([Token(text=token) for token in sentence.split() if token != ' '])
                    for sentence in re.split(sentence_split_regex, input_document_str)]
        else:
            nlp = spacy.load("en_core_web_sm") if lan_model is None else lan_model
            # preprocessed_documents = []
            disable_list = ['parser']
            if not nlp.has_pipe('sentencizer'):
                nlp.add_pipe(nlp.create_pipe('sentencizer'))
            # entities_of_documents = []
            # nested_sentences = nested_sentences_spacy_generator(nlp, texts, disable_list)

            chunking = True
            if chunking:
                chunked_texts, chunk_list = Preprocesser.chunk_text([input_document_str], 10000)

                logging.info(f'Start annotation of {len(chunked_texts)} chunked texts')
                nested_sentences = [[Sentence([token_spacy_representation(token)
                                               for token in sent if token.text != ' '])
                                     for sent in doc.sents]
                                    for doc in tqdm(nlp.pipe(chunked_texts, batch_size=50,
                                                             disable=disable_list),
                                                    desc="spacify",
                                                    total=len(chunked_texts),
                                                    disable=True,
                                                    )]
                nested_sentences = Preprocesser.merge_chunks(nested_sentences, chunk_list)

            else:
                logging.info(f'Start annotation of {len(input_document_str)} character sized text')
                nested_sentences = [[Sentence([token_spacy_representation(token)
                                               for token in sent if token.text != ' '])
                                     for sent in doc.sents]
                                    for doc in nlp.pipe([input_document_str], batch_size=100,
                                                        disable=disable_list)]
            if len(nested_sentences) > 1:
                raise UserWarning("Too many docs sentences")
            return nested_sentences[0]

    @staticmethod
    def filter_on_copy(corpus: Corpus,
                       pos: list = None,
                       remove_stopwords: bool = False,
                       remove_punctuation: bool = True,
                       remove_ne: bool = False,
                       masking: bool = False,
                       revert: bool = False) -> Corpus:

        def filter_condition(token: Token):
            if revert:
                return (not remove_stopwords or token.stop) \
                       and (not remove_punctuation or not token.alpha) \
                       and (not pos or token.pos in pos) \
                       and (not remove_ne or token.ne)
            else:
                return (not remove_stopwords or not token.stop) \
                       and (not remove_punctuation or token.alpha) \
                       and not (pos and token.pos in pos) \
                       and (not remove_ne or not token.ne)

        def mask(token: Token):
            new_token = Token(text=token.text,
                              lemma=token.lemma,
                              pos=token.pos,
                              ne=token.ne,
                              punctuation=token.punctuation,
                              alpha=token.alpha,
                              stop=token.stop)
            if not filter_condition(token):
                new_token.text = "del"
                new_token.lemma = "del"
            return new_token

        documents = {}
        for doc_id, document in corpus.documents.items():
            if not masking:
                new_sents = [Sentence([token for token in sentence.tokens if filter_condition(token)])
                             for sentence in document.sentences]
            else:
                new_sents = [Sentence([mask(token) for token in sentence.tokens])
                             for sentence in document.sentences]
            # print(new_sents)
            # document.set_sentences(new_sents)
            documents[doc_id] = Document(doc_id=doc_id,
                                         text=document.text,
                                         title=document.title,
                                         language=document.language,
                                         authors=document.authors,
                                         date=document.date,
                                         genres=document.genres,
                                         sentences=new_sents)

        new_corpus = Corpus(documents, corpus.name, corpus.language)

        new_corpus.set_series_dict(corpus.series_dict)
        new_corpus.set_document_entities()

        return new_corpus

    @staticmethod
    def filter_on_copy_mem_eff(corpus: Corpus,
                               corpus_dir: str,
                               pos: list = None,
                               remove_stopwords: bool = False,
                               remove_punctuation: bool = True,
                               remove_ne: bool = False,
                               masking: bool = False,
                               revert: bool = False) -> Corpus:

        def filter_condition(token: Token):
            if revert:
                return (not remove_stopwords or token.stop) \
                       and (not remove_punctuation or not token.alpha) \
                       and (not pos or token.pos in pos) \
                       and (not remove_ne or token.ne)
            else:
                return (not remove_stopwords or not token.stop) \
                       and (not remove_punctuation or token.alpha) \
                       and not (pos and token.pos in pos) \
                       and (not remove_ne or not token.ne)

        def mask(token: Token):
            new_token = Token(text=token.text,
                              lemma=token.lemma,
                              pos=token.pos,
                              ne=token.ne,
                              punctuation=token.punctuation,
                              alpha=token.alpha,
                              stop=token.stop)
            if not filter_condition(token):
                new_token.text = "del"
                new_token.lemma = "del"
            return new_token

        documents = {}
        file_dict = {}
        for doc_id, document in corpus.documents.items():
            if not masking:
                new_sents_gen = (Sentence([token for token in sentence.tokens if filter_condition(token)])
                                 for sentence in document.get_sentences_from_disk())
                new_sents = []
                for sent in new_sents_gen:
                    if len(sent.tokens) == 0:
                        sent.tokens.append(Token.empty_token())
                    new_sents.append(sent)
            else:
                new_sents = [Sentence([mask(token) for token in sentence.tokens])
                             for sentence in document.get_sentences_from_disk()]

            new_document = Document(doc_id=doc_id,
                                    text="",
                                    title=document.title,
                                    language=document.language,
                                    authors=document.authors,
                                    date=document.date,
                                    genres=document.genres,
                                    sentences=new_sents
                                    )

            new_document.calculate_sizes()
            # new_document.set_entities()

            new_document.file_path = os.path.join(corpus_dir, f'{new_document.meta_string_representation()}.txt')
            file_dict[doc_id] = new_document.file_path
            new_document.store_to_corpus_file(corpus_dir)

            new_document.sentences = None
            del new_sents
            documents[doc_id] = new_document

        new_corpus = Corpus(documents, corpus.name, corpus.language)
        new_corpus.set_series_dict(corpus.series_dict)
        # new_corpus.set_document_entities()

        new_corpus.file_dict = file_dict
        new_corpus.save_corpus_meta(corpus_dir)

        return new_corpus

    # @staticmethod
    # def filter(corpus: Corpus,
    #            pos: list = None,
    #            remove_stopwords: bool = False,
    #            remove_punctuation: bool = True,
    #            remove_ne: bool = False,
    #            masking: bool = False,
    #            revert: bool = False) -> Corpus:
    #
    #     def filter_condition(token: Token):
    #         if revert:
    #             return (not remove_stopwords or token.stop) \
    #                    and (not remove_punctuation or not token.alpha) \
    #                    and (not pos or token.pos in pos) \
    #                    and (not remove_ne or token.ne)
    #         else:
    #             return (not remove_stopwords or not token.stop) \
    #                    and (not remove_punctuation or token.alpha) \
    #                    and not (pos and token.pos in pos) \
    #                    and (not remove_ne or not token.ne)
    #
    #     def mask(token: Token):
    #         if not filter_condition(token):
    #             token.text = "del"
    #             token.lemma = "del"
    #         return token
    #
    #     for doc_id, document in corpus.documents.items():
    #         if not masking:
    #             new_sents = [Sentence([token for token in sentence.tokens if filter_condition(token)])
    #                          for sentence in document.sentences]
    #         else:
    #             new_sents = [Sentence([mask(token) for token in sentence.tokens])
    #                          for sentence in document.sentences]
    #
    #         document.set_sentences(new_sents)
    #     return corpus

    @staticmethod
    def filter_too_small_docs_from_corpus(corpus: Corpus, smaller_as: int = 20):
        documents = {doc_id: document
                     for doc_id, document in corpus.documents.items()
                     if document.sentences_nr >= smaller_as}
        new_corpus = Corpus(documents, name=corpus.name, language=corpus.language)
        # new_corpus.set_document_entities()

        return new_corpus


class CommonWords:
    @staticmethod
    def testing(dict_a: Dict[str, Set[str]], dict_b: Dict[str, Set[str]]):
        assert dict_a.keys() == dict_b.keys()
        for series_id in dict_a:
            set_a = dict_a[series_id]
            set_b = dict_b[series_id]
            assert set_a == set_b

    @staticmethod
    def without_gerneral_words(common_words: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        medium_common_words = {}
        for series_id_a, series_words_a in common_words.items():
            series_words_a_copy = set()
            series_words_a_copy.update(series_words_a)
            general_words = set()
            for series_id_b, series_words_b in common_words.items():
                if series_id_a != series_id_b:
                    general_words.update(series_words_a_copy.intersection(series_words_b))

            series_words_a_copy.difference_update(general_words)
            medium_common_words[series_id_a] = series_words_a_copy
        return medium_common_words

    @staticmethod
    def strict(series_dictionary: Dict[str, List[str]], doc_texts: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        common_words = defaultdict(set)
        for series_id, doc_ids in series_dictionary.items():
            series_words = []
            for doc_id in doc_ids:
                series_words.append(set(doc_texts[doc_id]))

            for token_set_a in series_words:
                for token_set_b in series_words:
                    if token_set_a != token_set_b:
                        common_words[series_id].update(token_set_a.intersection(token_set_b))
        return dict(common_words)

    @staticmethod
    def strict_general_words_sensitive(series_dictionary: Dict[str, List[str]], doc_texts: Dict[str, List[str]]) \
            -> Dict[str, Set[str]]:
        common_words = CommonWords.strict(series_dictionary, doc_texts)
        medium_common_words = CommonWords.without_gerneral_words(common_words)
        return medium_common_words

    @staticmethod
    def relaxed(series_dictionary: Dict[str, List[str]], doc_texts: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        common_words = defaultdict(set)
        for series_id, doc_ids in series_dictionary.items():
            series_words = []
            for doc_id in doc_ids:
                series_words.append(set(doc_texts[doc_id]))
            common_words[series_id] = set.intersection(*series_words)

        return dict(common_words)

    @staticmethod
    def relaxed_general_words_sensitive(series_dictionary: Dict[str, List[str]], doc_texts: Dict[str, List[str]]) \
            -> Dict[str, Set[str]]:
        common_words = CommonWords.relaxed(series_dictionary, doc_texts)
        medium_common_words = CommonWords.without_gerneral_words(common_words)
        return medium_common_words

    # @staticmethod
    # def global_common_words(doc_texts: Dict[str, List[str]]) -> Set[str]:
    #     tokens = [set(doc_tokens) for doc_id, doc_tokens in doc_texts.items()]
    #
    #     global_intersect = set()
    #     for token_set_a in tokens:
    #         for token_set_b in tokens:
    #             if token_set_a != token_set_b:
    #                 global_intersect.update(token_set_a.intersection(token_set_b))
    #
    #     global__strict_intersect = set.intersection(*tokens)
    #
    #     common_words = {}
    #     for c, (doc_id, doc_tokens) in enumerate(doc_texts.items()):
    #         not_to_delete = set(doc_tokens).difference(global_intersect).union(global__strict_intersect)
    #         to_delete = set(doc_tokens).difference(not_to_delete)
    #         common_words[doc_id] = to_delete
    #
    #     return common_words



    @staticmethod
    def global_too_specific_words_doc_frequency(corpus: Corpus, percentage_share: float = None,
                                                absolute_share: int = None) \
            -> Set[str]:
        tqdm_disable = True
        freq_dict = defaultdict(lambda: 0.0)

        if absolute_share:
            percentage_share = absolute_share
        print("Percantge share", percentage_share)
        for doc_id, document in tqdm(corpus.documents.items(), total=len(corpus.documents), desc="Calculate DF",
                                     disable=tqdm_disable):
            document: Document
            for token in set(document.get_flat_tokens_from_disk()):
                # vocab.add(token)
                if absolute_share:
                    freq_dict[token] += 1
                else:
                    freq_dict[token] += 1 / len(corpus.documents)
                    if freq_dict[token] > 1:
                        freq_dict[token] = 1

                    # if not lower_bound:
        #     lower_bound = lower_bound_absolute / len(doc_texts)
        # print(len(freq_dict.keys()), len(vocab))
        # print(min(freq_dict.values()), max(freq_dict.values()))
        to_remove = set([token for token, doc_freq in tqdm(freq_dict.items(), total=len(freq_dict),
                                                           desc="Filter DF",
                                                           disable=tqdm_disable)
                         if doc_freq <= percentage_share])
        # print(len(to_remove))
        # print(to_remove)
        # too_specific_words = {doc_id: set(doc_tokens).intersection(to_remove)
        #                       for doc_id, doc_tokens in tqdm(doc_texts.items(), total=len(doc_texts),
        #                                                      desc="Extract words to remove", disable=tqdm_disable)}
        # new_toks = set()
        # for doc_id, toks in too_specific_words.items():
        #     new_toks.update(toks)
        # print(len(new_toks))
        # opposite
        # too_specific_words = {doc_id: set(doc_tokens).difference(to_remove)
        #                 for doc_id, doc_tokens in doc_texts.items()}
        # print('>', too_specific_words)
        return to_remove
