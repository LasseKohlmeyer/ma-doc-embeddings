import json
import os
import re
from collections import defaultdict
from enum import Enum
import random
from typing import Union, List, Dict, Tuple, Set
import pandas as pd
from tqdm import tqdm
import spacy
from os import listdir
from os.path import isfile, join
import logging


class ConfigLoader:
    @staticmethod
    def get_config(relative_path=""):

        path = os.path.join(relative_path, "configs", "config.json")
        if os.path.exists(path):
            logging.info('importing config from configs/config.json ...')
            with open(path) as json_file:
                return json.load(json_file)

        path = os.path.join(relative_path, "default.config.json")
        if os.path.exists(path):
            path = os.path.join(relative_path, "configs", "default.config.json")
            logging.info('importing config from configs/default.config.json ...')
            with open(path) as json_file:
                return json.load(json_file)

        raise Exception("config file missing!")


config = ConfigLoader.get_config()


class Utils:
    @staticmethod
    def revert_dictionary(dictionary: Dict[Union[str, int], Union[str, int]]) -> Dict:
        d = defaultdict(list)
        for key, value in dictionary.items():
            d[value].append(key)

        return d

    @staticmethod
    def revert_dictionaried_list(dictionary: Dict[str, List[str]]):
        return {value: key for key, values in dictionary.items() for value in values}

    @staticmethod
    def revert_dictionaries_list(list_of_dictionaries: List[Dict[Union[str, int], Union[str, int]]]) -> List[Dict]:
        resulting_list = []
        for dictionary in list_of_dictionaries:
            resulting_list.append(Utils.revert_dictionary(dictionary))

        return resulting_list

    @staticmethod
    def revert_dictionaries_dict(list_of_dictionaries: Dict[str, Dict[Union[str, int], Union[str, int]]]) \
            -> Dict[str, Dict]:
        resulting_list = {}
        for key, dictionary in list_of_dictionaries.items():
            resulting_list[key] = Utils.revert_dictionary(dictionary)

        return resulting_list

    @staticmethod
    def revert_dictionaries(collection_of_dictionaries: Union[List[Dict[Union[str, int], Union[str, int]]],
                                                              Dict[str, Dict[Union[str, int], Union[str, int]]]]) \
            -> Union[List[Dict], Dict[str, Dict]]:
        if isinstance(collection_of_dictionaries, list):
            return Utils.revert_dictionaries_list(collection_of_dictionaries)
        elif isinstance(collection_of_dictionaries, dict):
            return Utils.revert_dictionaries_dict(collection_of_dictionaries)
        else:
            raise UserWarning("Passed entities are neither in list or dict!")


class DataHandler:
    @staticmethod
    def build_config_str(number_of_subparts: int, size: int, dataset: str, filter_mode: str,
                         vectorization_algorithm: str, fake_series: str):
        return f'{dataset}_{number_of_subparts}_{size}_{filter_mode}_{fake_series}_{vectorization_algorithm}'

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
        else:
            raise UserWarning("Unknown input string!")

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
            documents[doc_id] = Document(doc_id=doc_id,
                                         text=row["TEXT"],
                                         title=row["TITLE"],
                                         language=Language.EN,
                                         authors=row["AUTHORS"],
                                         date=row["DATE"],
                                         genres=genres)

        return Corpus(source=documents, name="book_summaries", language=Language.EN)

    @staticmethod
    def load_german_books_as_corpus(path: str = None) -> "Corpus":
        def load_textfile_book(prefix_path, suffix_path, document_id):
            with open(join(prefix_path, suffix_path), "r", encoding="utf-8") as file:
                content = file.read().replace('\n', ' ').replace('  ', ' ').replace('  ', ' ')
                # print(content)
                meta = suffix_path.replace('a.txt', '').replace('(', '').replace(')', '').split('_-_')
                author = meta[0].replace('_', ' ')
                title_year = ''.join(meta[1:]).replace('_', ' ')
                title = title_year[:-4]
                try:
                    year = int(title_year[-4:])
                except ValueError:
                    title = title_year
                    year = None
                # print(author, '|', title, '|', year)
                d = Document(doc_id=document_id,
                             text=content,
                             title=title,
                             language=Language.DE,
                             authors=author,
                             date=str(year),
                             genres=None)
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
            with open(join(prefix_path, suffix_path), "r", encoding="utf-8") as file:
                content = file.read().replace('\n', ' ').replace('  ', ' ').replace('  ', ' ')
                # print(content)
                meta = suffix_path.replace('a.txt', '').replace('(', '').replace(')', '').split('_-_')
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

                # print(author, '|', title, '|', year)
                d = Document(doc_id=document_id,
                             text=content,
                             title=title,
                             language=Language.DE,
                             authors=author,
                             date=str(year),
                             genres=None)
            return d

        if path is None:
            path = config["data_set_path"]["german_books"]

        path_a = join(path, 'corpus-of-german-fiction-txt')
        path_b = join(path, 'corpus-of-translated-foreign-language-fiction-txt')
        german_fiction = [f for f in listdir(path_a) if isfile(join(path_a, f))]
        tanslated_fiction = [f for f in listdir(path_b) if isfile(join(path_b, f))]

        series_paths = [(path_a, path) for path in german_fiction if "band" in path.lower()]
        series_paths.extend([(path_b, path) for path in tanslated_fiction if "band" in path.lower()])

        documents = {}
        for i, path in enumerate(series_paths):
            p_dir, p_file = path
            doc_id = f'sgf_{i}'
            documents[doc_id] = load_textfile_book(p_dir, p_file, doc_id)

        suffix_dict = defaultdict(list)
        for doc_id, document in documents.items():
            splitted_title = document.title.split('Band')
            band_title = splitted_title[0]
            if band_title == "Robin der Rote der ":
                band_title = "Robin der Rote "
            suffix_dict[band_title].append(doc_id)
        series_dict = {series.strip(): doc_ids for series, doc_ids in suffix_dict.items() if len(doc_ids) > 1}

        relevant_ids = [doc_id for series, doc_ids in series_dict.items() for doc_id in doc_ids]
        documents = {doc_id: document for doc_id, document in documents.items() if doc_id in relevant_ids}

        series_documents = {}
        new_series_dict = defaultdict(list)
        for index, (series, doc_ids) in enumerate(series_dict.items()):
            for doc_id in doc_ids:
                series_doc = documents[doc_id]
                series_id = int(series_doc.title.split()[-1]) - 1
                new_doc_id = f'gs_{index}_{series_id}'
                series_doc.doc_id = new_doc_id
                series_doc.title = series_doc.title.strip()
                series_documents[new_doc_id] = series_doc
                new_series_dict[f'gs_{index}'].append(new_doc_id)

        corpus = Corpus(source=series_documents, name="german_series", language=Language.DE)
        corpus.set_series_dict(new_series_dict)

        return corpus

    @staticmethod
    def load_tagged_german_books_as_corpus(path: str = None) -> "Corpus":
        def load_textfile_book(prefix_path, suffix_path, document_id):
            with open(join(prefix_path, suffix_path), "r", encoding="utf-8") as file:
                lines = file.read().split('\n')
                sentences = []
                tokens = []
                for line in lines:
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
                            logging.error(f'Error at {suffix_path}')

                content = ' '.join([' '.join(sentence.representation()) for sentence in sentences])
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
                             genres=None)
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
    def load_litrec_books_as_corpus(corpus_dir: str = None) -> "Corpus":
        def load_textfile_book(prefix_path, suffix_path, document_id, title):
            with open(join(prefix_path, suffix_path), "r", encoding="utf-8") as file:
                try:
                    content = file.read().replace('\n@\n', ' ').replace('\n', ' ').replace('  ', ' ').replace('  ', ' ')
                    content = ' '.join([token.split('/')[0] for token in content.split()])
                    # print(content)
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
                                 genres=None)
                except FileNotFoundError:
                    raise FileNotFoundError
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
    # E:/Corpora/LitRec-v1/books-v11\\Basil+King-14393-14393-8.txt.clean.pos
    # E:/Corpora/LitRec-v1/books-v11\\Basil+King-14394-14394-8.txt.clean.pos
    # @staticmethod
    # def path_exists(path: str) -> bool:
    #     return os.path.isfile(path)
    #
    # @staticmethod
    # def lines_from_file(path: str, encoding="utf-8") -> List[str]:
    #     with open(path, encoding=encoding) as f:
    #         data = f.read()
    #
    #     return data.split("\n")
    #
    # @staticmethod
    # def concat_path_sentences(paths: List[str]) -> List[str]:
    #     sentences = []
    #     for path in paths:
    #         sentences.extend(DataHandler.lines_from_file(path))
    #     return sentences
    #
    # @staticmethod
    # def load_folder_textfiles(directory: str) -> str:
    #     files = []
    #     for (dirpath, dirnames, filenames) in os.walk(directory):
    #         files.extend(filenames)
    #         break
    #
    #     for f in files:
    #         if f.startswith('.') or not f[-4:] == 'a.txt':
    #             files.remove(f)
    #
    #     docs = []
    #     for file in tqdm(files, desc=f"load_keyed_vecs documents in {directory}", total=len(files)):
    #         docs.append(" ".join(DataHandler.lines_from_file(os.path.join(directory, file))).strip())
    #     return " ".join(docs)
    #
    # @staticmethod
    # def sentenize(document_text):
    #     german_model = spacy.load("de_core_news_sm")
    #     sbd = german_model.create_pipe('sentencizer')
    #     german_model.add_pipe(sbd)
    #
    #     doc = german_model(document_text)
    #
    #     sents_list = []
    #     for sent in doc.sents:
    #         sents_list.append(sent.text)
    #
    #     return sents_list
    #
    # @staticmethod
    # def save(file_path, content):
    #     with open(file_path, 'w', encoding="utf-8") as the_file:
    #         the_file.write(content)
    #
    # @staticmethod
    # def read_files_and_save_sentences_to_dir(directory: str):
    #     sentences = DataHandler.sentenize(DataHandler.load_folder_textfiles(directory))
    #     DataHandler.save(file_path=os.path.join(directory, "all_sentences.txt"), content="\n".join(sentences))
    #
    # @staticmethod
    # def split_data(file_path: str, lines_per_file: int, new_name: str):
    #     lines = DataHandler.lines_from_file(file_path)
    #     total_lines = len(lines)
    #     i = 0
    #     while True:
    #         if i * lines_per_file > total_lines:
    #             break
    #         DataHandler.save(f'{new_name}_{i}a.txt', "\n".join(lines[i * lines_per_file: (i + 1) * lines_per_file]))
    #         i += 1
    #
    # @staticmethod
    # def tidy_indices(file_path, new_path):
    #     lines = DataHandler.lines_from_file(file_path)
    #     lines = ["\t".join(line.split('\t')[1:]) for line in lines]
    #     DataHandler.save(new_path, "\n".join(lines))


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


class Token:
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

    def representation(self, lemma: bool = False, lower: bool = False):
        if lemma:
            rep = self.lemma
        else:
            rep = self.text
        if lower:
            rep = rep.lower()
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

    def get_save_file_representation(self):
        def bool_converter(input_bool: bool) -> str:
            if input_bool:
                return "1"
            else:
                return "0"

        return f'{self.text }\t{self.lemma}\t{str(self.pos).strip()}\t{str(self.ne).strip()}' \
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
    # def __init__(self, tokens: List[str]):
    #     self.tokens = tokens
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens

    def representation(self, lemma: bool = False, lower: bool = False):
        return [token.representation(lemma, lower) for token in self.tokens]

    def json_representation(self):
        return vars(self)

    def __str__(self):
        return str(self.representation())

    __repr__ = __str__


class Document:
    def __init__(self, doc_id: str, text: str, title: str, language: Language,
                 authors: str = None, date: str = None, genres: str = None, sentences: List[Sentence] = None):
        self.doc_id = doc_id
        self.text = text
        self.title = title
        self.language = language
        self.authors = authors
        self.date = date
        self.genres = genres
        if sentences is None:
            sentences = []
        self.sentences: List[Sentence] = sentences  # None
        self.absolute_positions = {}
        # self.tokens: List[str] = []  # None

    # def set_sentences(self, sentences: List[List[str]]):
    #     self.sentences = [Sentence(sentence) for sentence in sentences]
    #     self.tokens = [token for sentence in sentences for token in sentence]

    def set_sentences(self, sentences: List[Sentence]):
        self.sentences = sentences
        # self.tokens = [token for sentence in sentences for token in sentence.tokens]

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

    def get_flat_document_tokens(self, lemma: bool = False, lower: bool = False):
        # for doc_id, document in self.documents.items():
        #     print(document.sentences[1].tokens)
        # print('>', self.sentences)
        tokens = [token.representation(lemma, lower) for sentence in self.sentences for token in sentence.tokens]
        if len(tokens) == 0:
            raise UserWarning("No sentences set")
        return tokens

    def __str__(self):
        return f'{self.authors} ({self.date}): {self.title[:50]}'

    __repr__ = __str__

    def json_representation(self):
        return vars(self)

    def meta_string_representation(self):
        pattern = re.compile(r'[\W]+', re.UNICODE)
        return f'{self.doc_id}_-_{str(self.authors).replace(" ", "_")}_-_' \
               f'{pattern.sub("", str(self.title)).replace(" ", "_")}_-_' \
               f'{self.language}_-_{str(self.genres).replace(" ", "_")}_-_{self.date}'

    @staticmethod
    def create_document_from_doc_file(doc_path: str):
        with open(doc_path, "r", encoding="utf-8") as file:
            lines = file.read().split('\n')
            sentences = []
            tokens = []
            for line in lines:
                try:
                    tokens.append(Token.parse_text_file_token_representation(line))
                except ValueError:
                    if line == '<SENT>':
                        sentences.append(Sentence(tokens))
                        tokens = []
                    elif line == '' or line is None:
                        # skip line
                        pass
                    else:
                        logging.error(f'Error at {doc_path}')

            fn = os.path.basename(doc_path)
            doc_id, authors, title, language, genres, date = fn.replace('.txt', '').split('_-_')
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
            text = ' '.join([' '.join(sentence.representation()) for sentence in sentences])
            return Document(doc_id=doc_id, text=text, title=title, language=Language.get_from_str(language),
                            authors=authors, date=date, genres=genres, sentences=sentences)


class Corpus:
    def __init__(self, source: Union[Dict[Union[str, int], Document], List[Document], str],
                 name: str = None,
                 language: Language = None):
        self.name = name
        self.language = language
        self.document_entities = None
        self.series_dict = None
        self.root_corpus_path = None
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

    def save_corpus_adv(self, corpus_dir: str):
        if not os.path.isdir(corpus_dir):
            os.mkdir(corpus_dir)
        for doc_id, document in self.documents.items():
            document.meta_string_representation()
            doc_path = os.path.join(corpus_dir, f'{document.meta_string_representation()}.txt')
            with open(doc_path, 'w', encoding="utf-8") as writer:
                for sentence in document.sentences:
                    for token in sentence.tokens:
                        writer.write(f'{token.get_save_file_representation()}\n')
                    writer.write("<SENT>\n")
        if self.root_corpus_path is None:
            self.root_corpus_path = corpus_dir
        data = {"name": self.name, "root_corpus_path": self.root_corpus_path,
                "language": self.language, "series_dict": self.series_dict}
        with open(os.path.join(corpus_dir, "meta_info.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1, default=lambda o: o.__dict__)

    @staticmethod
    def load_corpus_from_dir_format(corpus_dir: str):
        meta_path = os.path.join(corpus_dir, "meta_info.json")
        with open(meta_path, 'r', encoding='utf-8') as file:
            meta_data = json.loads(file.read())

        document_paths = [file_path for file_path in os.listdir(corpus_dir) if file_path.endswith('.txt')]
        documents = [Document.create_document_from_doc_file(os.path.join(corpus_dir, doc_path))
                     for doc_path in document_paths]
        corpus = Corpus(source=documents, name=meta_data["name"], language=meta_data["language"])
        corpus.root_corpus_path = meta_data["root_corpus_path"]
        corpus.set_series_dict(meta_data["series_dict"])
        return corpus

    @staticmethod
    def fast_load(number_of_subparts=None, size=None, data_set=None, filer_mode=None, fake_real=None, path=None):
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
        else:
            if os.path.exists(path):
                corpus = Corpus.load_corpus_from_dir_format(path)
            else:
                corpus = Corpus(f'{path}.json')
                corpus.save_corpus_adv(path)
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

    def token_number(self):
        c = 0
        for d in self.get_documents():
            c += len(d.text.split())
        return c

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

    def get_texts_and_doc_ids(self):
        return map(list, zip(*[(document.text, doc_id) for doc_id, document in self.documents.items()]))

    def id2desc(self, index: Union[str, int]):
        if index.endswith('_sum'):
            index = index.replace('_sum', '')
        elif index.endswith('_time'):
            index = index.replace('_time', '')
        elif index.endswith('_loc'):
            index = index.replace('_loc', '')
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
            doc: Document
            doc_entities = defaultdict(list)
            for sent in doc.sentences:
                for token in sent.tokens:
                    if token.ne:
                        # print(token.ne, token.text)
                        doc_entities[token.ne].append(token)
            entities_dict[doc_id] = doc_entities
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
                token_reprs = [token.representation() for sentence in self.documents[doc_id].sentences
                               for token in sentence.tokens]
                for time_entity in time_entities:
                    tm = time_entity.split(' ')
                    positions = find_sub_list(tm, token_reprs)
                    for position in positions:
                        start, end = position
                        for i in range(start, end+1):
                            self.documents[doc_id].get_token_at_doc_position(i).ne = "TIME"

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

    # def set_root_path(self, root_path: str):
    #     self.root_corpus_path = root_path

    def get_flat_document_tokens(self, lemma: bool = False, lower: bool = False):
        # for doc_id, document in self.documents.items():
        #     print(document.sentences[1].tokens)

        tokens = [[token.representation(lemma, lower) for sentence in document.sentences for token in sentence.tokens]
                  for doc_id, document in self.documents.items()]
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

        if len(tokens) == 0:
            raise UserWarning("No sentences set")

        return tokens

    def get_flat_corpus_sentences(self, lemma: bool = False, lower: bool = False):
        sentences = [sentence.representation(lemma, lower)
                     for doc_id, document in self.documents.items()
                     for sentence in document.sentences]
        if len(sentences) == 0:
            raise UserWarning("No sentences set")
        return sentences

    def fake_series(self, number_of_sub_parts=2) -> Tuple["Corpus", Dict[str, List[str]]]:
        fake_series_corpus = []
        fake_series_dict = defaultdict(list)

        for doc_id, document in self.documents.items():
            # print(len(document.sentences), number_of_sub_parts)
            if len(document.sentences) < number_of_sub_parts:
                raise UserWarning("Nr of document sentences too small!")
            sentence_counter = 0
            # avg_doc_length = math.ceil(len(document.sentences) / number_of_sub_parts)
            avg_doc_length = len(document.sentences) // number_of_sub_parts
            # print(doc_id, len(document.sentences))
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
                sub_sentences = document.sentences[i * avg_doc_length:end]
                fake_series_doc.set_sentences(sub_sentences)
                fake_series_doc.reset_text_based_on_sentences()
                # if len(fake_series_doc.sentences) == 0:
                #     print(document.date, document.doc_id, document.title, fake_series_doc.text, document.text)
                #     print(sentence_counter, len(document.sentences), avg_doc_length)
                fake_series_corpus.append(fake_series_doc)
                sentence_counter += len(fake_series_doc.sentences)

                assert len(fake_series_doc.sentences) > 0
            assert sentence_counter == len(document.sentences)
        fake_series_corpus = Corpus(fake_series_corpus, name=f'{self.name}_fake', language=self.language)
        fake_series_corpus.set_document_entities()
        fake_series_corpus.set_series_dict(fake_series_dict)
        # for doc_id, doc in fake_series_corpus.documents.items():
        #     print(doc_id, doc.text)
        # print(fake_series_corpus)
        return fake_series_corpus, fake_series_dict

    def get_common_words(self, series_dictionary: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        common_words = defaultdict(set)
        for series_id, doc_ids in series_dictionary.items():
            series_words = []
            for doc_id in doc_ids:
                series_words.append(set(self.documents[doc_id].get_flat_document_tokens(lemma=True, lower=True)))

            common_words[series_id] = set.intersection(*series_words)
            for doc_id in doc_ids:
                common_words[doc_id] = common_words[series_id]
        return common_words

    def filter(self, mode: str, masking: bool = False, common_words: Dict[str, Set[str]] = None):
        def filter_condition(token: Token, document_id: str):
            # print(token.representation(lemma=True, lower=True) not in common_words[document_id],
            #       token.representation(lemma=True, lower=True),
            #       common_words[document_id])
            return token.representation(lemma=True, lower=True) not in common_words[document_id]

        def mask(token: Token, document_id: str):
            if not filter_condition(token, document_id):
                token.text = "del"
                token.lemma = "del"
            return token

        if mode.lower() == "no_filter" or mode.lower() == "nf":
            pass
            # return self
        elif mode.lower() == "common_words" or mode.lower() == "cw":
            # print('>>', common_words["bs_0"])
            for doc_id, document in tqdm(self.documents.items(), total=len(self.documents), desc="Filtering corpus",
                                         disable=True):
                if not masking:
                    new_sents = [Sentence([token for token in sentence.tokens if filter_condition(token, doc_id)])
                                 for sentence in document.sentences]
                    for sent in new_sents:
                        if len(sent.tokens) == 0:
                            sent.tokens.append(Token.empty_token())
                    # for new_s in new_sents:
                    #     print(new_s.representation())
                else:
                    new_sents = [Sentence([mask(token, doc_id) for token in sentence.tokens])
                                 for sentence in document.sentences]
                    # for new_s in new_sents:
                    #     print(new_s.representation())
                document.set_sentences(new_sents)
                # return self
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
                raise UserWarning("Not supported mode")
            Preprocesser.filter(self,
                                pos=pos,
                                remove_stopwords=remove_stopwords,
                                remove_punctuation=remove_punctuation,
                                remove_ne=remove_ne,
                                masking=masking,
                                revert=revert)

    def filter_on_copy(self, mode: str, masking: bool = False, common_words: Dict[str, Set[str]] = None) -> "Corpus":
        def filter_condition(token: Token, document_id: str):
            # print(token.representation(lemma=True, lower=True) not in common_words[document_id],
            #       token.representation(lemma=True, lower=True),
            #       common_words[document_id])
            return token.representation(lemma=True, lower=True) not in common_words[document_id]

        def mask(token: Token, document_id: str):
            if not filter_condition(token, document_id):
                token.text = "del"
                token.lemma = "del"
            return token

        if mode.lower() == "no_filter" or mode.lower() == "nf":
            return self
        elif mode.lower() == "common_words" or mode.lower() == "cw":
            # print('>>', common_words["bs_0"])
            documents = {}
            for doc_id, document in tqdm(self.documents.items(), total=len(self.documents), desc="Filtering corpus",
                                         disable=True):
                if not masking:
                    new_sents = [Sentence([token for token in sentence.tokens if filter_condition(token, doc_id)])
                                 for sentence in document.sentences]
                    for sent in new_sents:
                        if len(sent.tokens) == 0:
                            sent.tokens.append(Token.empty_token())
                    # for new_s in new_sents:
                    #     print(new_s.representation())
                else:
                    new_sents = [Sentence([mask(token, doc_id) for token in sentence.tokens])
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

            corpus = Corpus(documents, self.name, self.language)
            corpus.set_series_dict(self.series_dict)
            corpus.set_document_entities()

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

    def __iter__(self):
        return self.documents.values().__iter__()

    def __len__(self):
        return len(self.documents)

    def __str__(self):
        return f'docs={len(self)}, lan={self.language}, name={self.name}'

    def __getitem__(self, key):
        if isinstance(key, slice):
            # do your handling for a slice object:
            # print(key.start, key.stop, key.step)
            return Corpus(source=list(self.documents.values())[key.start: key.stop: key.step],
                          name=f'{self.name}_{key.start}_{key.stop}_{key.step}',
                          language=self.language)
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


class Preprocesser:
    @classmethod
    def tokenize(cls, text: Union[str, List[str]]):
        if isinstance(text, str):
            return text.split()
        else:
            return [cls.tokenize(text) for text in text]

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
    def merge_chunks(chunked_texts: List[List[Sentence]], chunked_list: List[bool]):
        unchunked = []
        unchunked_object = []

        for is_chunked, text in zip(chunked_list, chunked_texts):
            unchunked_object.extend(text)
            if not is_chunked:
                if isinstance(text, list):
                    unchunked.append(unchunked_object.copy())
                else:
                    raise UserWarning("Not supported type!")
                unchunked_object.clear()
        return unchunked

    @staticmethod
    def annotate_corpus(corpus: Corpus):
        texts, doc_ids = corpus.get_texts_and_doc_ids()
        lan_model = corpus.give_spacy_lan_model()
        prep_sentences = Preprocesser.annotate_tokens(texts,
                                                      doc_ids,
                                                      lan_model
                                                      )

        preprocessed_corpus = Corpus(corpus.documents,
                                     name=f'{corpus.name}_prep',
                                     language=corpus.language)
        preprocessed_corpus.set_sentences(prep_sentences)
        preprocessed_corpus.set_document_entities()
        preprocessed_corpus.set_series_dict(corpus.series_dict)

        return preprocessed_corpus

    @staticmethod
    def structure_string_texts(texts: List[str], lan_model, lemma: bool = False, lower: bool = False):
        prep_sentences = Preprocesser.annotate_tokens_list(texts, lan_model)

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
    def annotate_tokens_list(texts, lan_model=None) -> List[List[Sentence]]:
        def token_representation(token):
            return Token(text=token.text,
                         lemma=token.lemma_,
                         pos=token.pos_,
                         ne=token.ent_type_,
                         punctuation=token.is_punct,
                         alpha=token.is_alpha,
                         stop=token.is_stop)

        nlp = spacy.load("en_core_web_sm") if lan_model is None else lan_model
        # preprocessed_documents = []
        disable_list = ['parser']
        if not nlp.has_pipe('sentencizer'):
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
        # entities_of_documents = []
        nested_sentences = []
        chunked_texts, chunk_list = Preprocesser.chunk_text(texts, 5000)
        logging.info(f'Start annotation of {len(chunked_texts)} chunked texts')
        for doc in nlp.pipe(chunked_texts, disable=disable_list):
            # ents = {e.text: e.label_ for e in doc.ents}

            # entities_of_documents.append(ents)
            preprocessed_sentences = []
            # all_document_sentences = []
            for sent in doc.sents:
                sentence_tokens = Sentence([token_representation(token)
                                            for token in sent if token.text != ' '])
                preprocessed_sentences.append(sentence_tokens)
                # all_document_sentences.extend(sentence_tokens)

            # preprocessed_documents.append(all_document_sentences)
            nested_sentences.append(preprocessed_sentences)

        # print(len(nested_sentences))
        nested_sentences = Preprocesser.merge_chunks(nested_sentences, chunk_list)
        # print(len(nested_sentences))
        return nested_sentences

    @staticmethod
    def annotate_tokens(texts, doc_ids, lan_model=None):
        nested_sentences = Preprocesser.annotate_tokens_list(texts, lan_model)
        nested_sentences_dict = {doc_id: doc_sents for doc_id, doc_sents in zip(doc_ids, nested_sentences)}
        # entities_of_documents_dict = {doc_id: doc_ents for doc_id, doc_ents in zip(doc_ids, entities_of_documents)}
        # print(nested_sentences_dict)
        # print(entities_of_documents_dict)

        # for doc_id, d in nested_sentences_dict.items():
        #     print(doc_id, d[0])

        return nested_sentences_dict

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
    def filter(corpus: Corpus,
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
            if not filter_condition(token):
                token.text = "del"
                token.lemma = "del"
            return token

        for doc_id, document in corpus.documents.items():
            if not masking:
                new_sents = [Sentence([token for token in sentence.tokens if filter_condition(token)])
                             for sentence in document.sentences]
            else:
                new_sents = [Sentence([mask(token) for token in sentence.tokens])
                             for sentence in document.sentences]

            document.set_sentences(new_sents)
        return corpus

    @staticmethod
    def filter_too_small_docs_from_corpus(corpus: Corpus, smaller_as: int = 20):
        documents = {doc_id: document
                     for doc_id, document in corpus.documents.items()
                     if len(document.sentences) >= smaller_as}
        new_corpus = Corpus(documents, name=corpus.name, language=corpus.language)
        new_corpus.set_document_entities()

        return new_corpus
