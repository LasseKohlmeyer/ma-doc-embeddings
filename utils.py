import json
import logging
from collections import defaultdict
from enum import Enum
import random
from typing import Union, List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import spacy
from os import listdir
from os.path import isfile, join


class Utils:
    @staticmethod
    def revert_dictionary(dictionary: Dict[Union[str, int], Union[str, int]]) -> Dict:
        d = defaultdict(list)
        for key, value in dictionary.items():
            d[value].append(key)

        return d

    @staticmethod
    def revert_dictionaries_list(list_of_dictionaries: List[Dict[Union[str, int], Union[str, int]]]) -> List[Dict]:
        print(list_of_dictionaries)
        resulting_list = []
        for dictionary in list_of_dictionaries:
            resulting_list.append(Utils.revert_dictionary(dictionary))

        return resulting_list

    @staticmethod
    def revert_dictionaries_dict(list_of_dictionaries: Dict[str, Dict[Union[str, int], Union[str, int]]]) \
            -> Dict[str, Dict]:
        print(list_of_dictionaries)
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
    def load_book_summaries_as_corpus(path: str = None) -> "Corpus":
        if path is None:
            path = 'E:/Corpora/JRBookSummaries/booksummaries.txt'

        book_summary_df = pd.read_csv(path, delimiter='\t')

        # book_summary_df.columns = [['ID_A', 'ID_B', 'TITLE', 'AUTHORS', 'DATE', 'GENRES', 'TEXT']]
        print(book_summary_df[['GENRES']].head())

        documents = {}
        for i, row in tqdm(book_summary_df.iterrows(), total=len(book_summary_df.index), desc="Parse Documents"):
            doc_id = f'bs_{i}'
            documents[doc_id] = Document(doc_id=doc_id,
                                         text=row["TEXT"],
                                         title=row["TITLE"],
                                         language=Language.EN,
                                         authors=row["AUTHORS"],
                                         date=row["DATE"],
                                         genres=row["GENRES"])

        return Corpus(source=documents, name="book_summaries", language=Language.EN)

    @staticmethod
    def load_german_books_as_corpus(path: str = None) -> "Corpus":
        def load_textfile_book(prefix_path, suffix_path, document_id):
            with open(join(prefix_path, suffix_path), "r", encoding="utf-8") as file:
                content = file.read().replace('\n', ' ').replace('  ', ' ').replace('  ', ' ')
                # print(content)
                meta = suffix_path.replace('.txt', '').replace('(', '').replace(')', '').split('_-_')
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
            path = 'E:/Corpora/Corpus of German-Language Fiction'

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
    def load_litrec_books_as_corpus(corpus_dir: str = None) -> "Corpus":
        def load_textfile_book(prefix_path, suffix_path, document_id, title):
            with open(join(prefix_path, suffix_path), "r", encoding="utf-8") as file:
                try:
                    content = file.read().replace('\n@\n', ' ').replace('\n', ' ').replace('  ', ' ').replace('  ', ' ')
                    content = ' '.join([token.split('/')[0] for token in content.split()])
                    # print(content)
                    meta = suffix_path.replace('.txt.clean.pos', '').split('-')
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
            corpus_dir = 'E:/Corpora/LitRec-v1/'
        df = pd.read_csv(join(corpus_dir, 'user-ratings-v1.txt'), delimiter='@')

        documents = {}
        not_found = []

        filenames = df[['filename', 'title']].drop_duplicates()
        for i, row in tqdm(filenames.iterrows(), total=len(filenames.index)):
            doc_id = f'lr_{i}'
            try:
                documents[doc_id] = load_textfile_book(prefix_path=join(corpus_dir, 'books-v11'),
                                                       suffix_path=row['filename'],
                                                       document_id=doc_id,
                                                       title=row['title'])
            except FileNotFoundError:
                not_found.append((row['title'], row['filename']))
        print(len(not_found))
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
    #         if f.startswith('.') or not f[-4:] == '.txt':
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
    #         DataHandler.save(f'{new_name}_{i}.txt', "\n".join(lines[i * lines_per_file: (i + 1) * lines_per_file]))
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


class Sentence:
    # def __init__(self, tokens: List[str]):
    #     self.tokens = tokens
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens

    def representation(self, lemma: bool = False, lower: bool = False):
        return [token.representation(lemma, lower) for token in self.tokens]


class Document:
    def __init__(self, doc_id: str, text: str, title: str, language: Language,
                 authors: str = None, date: str = None, genres: str = None):
        self.doc_id = doc_id
        self.text = text
        self.title = title
        self.language = language
        self.authors = authors
        self.date = date
        self.genres = genres
        self.sentences: List[Sentence] = []  # None
        # self.tokens: List[str] = []  # None

    # def set_sentences(self, sentences: List[List[str]]):
    #     self.sentences = [Sentence(sentence) for sentence in sentences]
    #     self.tokens = [token for sentence in sentences for token in sentence]

    def set_sentences(self, sentences: List[Sentence]):
        self.sentences = sentences
        # self.tokens = [token for sentence in sentences for token in sentence.tokens]

    def __str__(self):
        return f'{self.authors} ({self.date}): {self.title[:50]}'

    __repr__ = __str__


class Corpus:
    def __init__(self, source: Union[Dict[Union[str, int], Document], List[Document], str],
                 name: str,
                 language: Language):
        self.name = name
        self.language = language
        self.document_entities = None

        if isinstance(source, str):
            documents = self.load_corpus(path=source)
        else:
            documents = source
        if isinstance(documents, dict):
            self.documents: Dict[str, Document] = documents
        elif isinstance(documents, list):
            self.documents: Dict[str, Document] = {document.doc_id: document for document in documents}
        else:
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
        data = [doc.__dict__ for doc in self.get_documents()]
        # data = {doc.doc_id: doc.__dict__ for doc in self.get_documents()}

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1, default=lambda o: o.__dict__)
        logging.info(f'saved {path}')

    def get_years(self) -> [str]:
        years = set()
        for d in self.get_documents(as_list=True):
            if d.date:
                years.add(d.date)
        return sorted(list(years))

    @staticmethod
    def load_corpus(path: str) -> List[Document]:
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

    def token_number(self):
        c = 0
        for d in self.get_documents():
            c += len(d.text.split())
        return c

    def year_wise(self, ids: bool = False) -> Dict[int, List[Union[str, int, Document]]]:
        year_bins = defaultdict(list)

        for doc in self.get_documents():
            if ids:
                year_bins[doc.date].append(doc.doc_id)
            else:
                year_bins[doc.date].append(doc)

        return year_bins

    def sample(self, number_documents=100, as_corpus=True, seed=None):
        if len(self) < number_documents:
            return self

        if seed:
            random.seed(seed)

        if as_corpus:
            result = Corpus(source=random.sample(self.get_documents(), k=number_documents),
                            language=self.language,
                            name=f'{self.name}_sample')
        else:
            result = random.sample(self.get_documents(), k=number_documents)
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
            print(f"Language {Language.EN} detected.")
            return spacy.load("en_core_web_sm")
        else:
            print(f"Language {Language.DE} detected.")
            return spacy.load("de_core_news_sm")

    def set_document_entities(self, entities_dict: Dict[str, List[str]]):
        self.document_entities = entities_dict

    def set_sentences(self, sentences: Dict[str, List[Sentence]]):
        for doc_id, document in self.documents.items():
            document.set_sentences(sentences[doc_id])

    def get_flat_document_tokens(self, lemma: bool = False, lower: bool = False):
        # for doc_id, document in self.documents.items():
        #     print(document.sentences[1].tokens)
        return [[token.representation(lemma, lower) for sentence in document.sentences for token in sentence.tokens]
                for doc_id, document in self.documents.items()]

    def get_flat_corpus_sentences(self, lemma: bool = False, lower: bool = False):
        return [sentence.representation(lemma, lower)
                for doc_id, document in self.documents.items()
                for sentence in document.sentences]

    def fake_series(self) -> Tuple["Corpus", Dict[str, List[str]]]:
        pass

    def get_common_words(self, series_dictionary: Dict[str, List[str]]) -> Dict[str, List[str]]:
        pass

    def filter(self, mode: str) -> "Corpus":
        pass

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
                          name=f'{self.name}_{key.start}:{key.stop}:{key.step}',
                          language=self.language)
        elif isinstance(key, int):
            # print(key)
            return list(self.documents.values())[key]
        else:
            # Do your handling for a plain index
            # print(key)
            return self.documents[key]

    __repr__ = __str__


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

    @staticmethod
    def merge_chunks(chunked_texts: Union[List[str], List[List[str]], List[Dict]], chunked_list: List[bool]):
        unchunked = []
        if all(isinstance(n, dict) for n in chunked_texts):
            dict_usage = True
            unchunked_object = {}
        else:
            dict_usage = False
            unchunked_object = []

        for is_chunked, text in zip(chunked_list, chunked_texts):
            if isinstance(text, str):
                unchunked_object.append(text)
            else:
                if dict_usage:
                    unchunked_object.update(text)
                else:
                    unchunked_object.extend(text)
            if not is_chunked:
                if isinstance(text, str):
                    unchunked.append(' '.join(unchunked_object))
                elif isinstance(text, list):
                    unchunked.append(unchunked_object.copy())
                elif isinstance(text, dict):
                    unchunked.append(unchunked_object.copy())
                else:
                    raise UserWarning("Not supported type!")

                unchunked_object.clear()
        return unchunked

    @staticmethod
    def annotate_corpus(corpus: Corpus):
        texts, doc_ids = corpus.get_texts_and_doc_ids()
        lan_model = corpus.give_spacy_lan_model()
        prep_sentences, prep_entities = Preprocesser.annotate_tokens(texts,
                                                                       doc_ids,
                                                                       # lemmatize,
                                                                       # lower,
                                                                       # remove_punctuation,
                                                                       lan_model
                                                                     )

        preprocessed_corpus = Corpus(corpus.documents,
                                     name=f'{corpus.name}_prep',
                                     language=corpus.language)
        preprocessed_corpus.set_sentences(prep_sentences)
        preprocessed_corpus.set_document_entities(prep_entities)

        return preprocessed_corpus

    @staticmethod
    def structure_string_texts(texts: List[str], lan_model, lemma: bool = False, lower: bool = False):
        prep_sentences, _ = Preprocesser.annotate_tokens_list(texts, lan_model)

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
    def annotate_tokens_list(texts, lan_model=None) -> Tuple[List[List[Sentence]], List[Dict]]:
        def token_representation(token):
            return Token(text=token.text,
                         lemma=token.lemma_,
                         pos=token.pos_,
                         ne=token.ent_type,
                         punctuation=token.is_punct,
                         alpha=token.is_alpha,
                         stop=token.is_stop)

        nlp = spacy.load("en_core_web_sm") if lan_model is None else lan_model
        # preprocessed_documents = []
        disable_list = ['parser']
        if not nlp.has_pipe('sentencizer'):
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
        entities_of_documents = []
        nested_sentences = []
        # chunked_texts, chunk_list = Preprocesser.chunk_text(texts, 5000)

        for doc in tqdm(nlp.pipe(texts, disable=disable_list), desc="Annotation", total=len(texts)):
            ents = {e.text: e.label_ for e in doc.ents}

            entities_of_documents.append(ents)
            preprocessed_sentences = []
            # all_document_sentences = []
            for sent in doc.sents:
                sentence_tokens = Sentence([token_representation(token)
                                            for token in sent if token.text != ' '])
                preprocessed_sentences.append(sentence_tokens)
                # all_document_sentences.extend(sentence_tokens)

            # preprocessed_documents.append(all_document_sentences)
            nested_sentences.append(preprocessed_sentences)

        return nested_sentences, entities_of_documents

    @staticmethod
    def annotate_tokens(texts, doc_ids, lan_model=None):
        nested_sentences, entities_of_documents = Preprocesser.annotate_tokens_list(texts, lan_model)
        nested_sentences_dict = {doc_id: doc_sents for doc_id, doc_sents in zip(doc_ids, nested_sentences)}
        entities_of_documents_dict = {doc_id: doc_ents for doc_id, doc_ents in zip(doc_ids, entities_of_documents)}
        # print(nested_sentences_dict)
        # print(entities_of_documents_dict)

        # for doc_id, d in nested_sentences_dict.items():
        #     print(doc_id, d[0])

        return nested_sentences_dict, entities_of_documents_dict

    @staticmethod
    def simple_preprocess(texts, doc_ids, lemmatize: bool = False, lower: bool = False, remove_punctuation: bool = True,
                          lan_model=None):
        def token_representation(token):
            representation = str(token.lemma_) if lemmatize else str(token)
            if lower:
                representation = representation.lower()
            return representation

        def filter_condition(token):
            return not remove_punctuation or token.is_alpha

        nlp = spacy.load("en_core_web_sm") if lan_model is None else lan_model


        preprocessed_documents = []
        disable_list = ['parser', 'tagger']

        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        entities_of_documents = []
        nested_sentences = []
        # chunked_texts, chunk_list = Preprocesser.chunk_text(texts, 5000)

        for doc in tqdm(nlp.pipe(texts, disable=disable_list), desc="Preprocessing", total=len(texts)):
            ents = {e.text: e.label_ for e in doc.ents}

            entities_of_documents.append(ents)
            preprocessed_sentences = []
            all_document_sentences = []
            for sent in doc.sents:
                sentence_tokens = [token_representation(token)
                                   for token in sent if filter_condition(token)]
                preprocessed_sentences.append(sentence_tokens)
                all_document_sentences.extend(sentence_tokens)

            preprocessed_documents.append(all_document_sentences)
            nested_sentences.append(preprocessed_sentences)

        nested_sentences_dict = {doc_id: doc_sents for doc_id, doc_sents in zip(doc_ids, nested_sentences)}
        entities_of_documents_dict = {doc_id: doc_ents for doc_id, doc_ents in zip(doc_ids, entities_of_documents)}
        # print(nested_sentences_dict)
        # print(entities_of_documents_dict)

        # for doc_id, d in nested_sentences_dict.items():
        #     print(doc_id, d[0])

        return nested_sentences_dict, entities_of_documents_dict

    @staticmethod
    def preprocess(texts, lemmatize: bool = False, lower: bool = False,
                   pos_filter: list = None, remove_stopwords: bool = False,
                   remove_punctuation: bool = True, lan_model=None,
                   ner: bool = True, return_in_sentence_format: bool = False):
        def token_representation(token):
            representation = str(token.lemma_) if lemmatize else str(token)
            if lower:
                representation = representation.lower()
            return representation

        def filter_condition(token):
            return (not remove_stopwords or not token.is_stop) \
                   and (not remove_punctuation or token.is_alpha) \
                   and (not pos_filter or token.pos_ in pos_filter) \
                   and (not ner or not token.ent_type)
            # and (not ner or token.text not in ents)

        nlp = spacy.load("en_core_web_sm") if lan_model is None else lan_model

        preprocessed_texts = []
        preprocessed_documents = []
        disable_list = ['parser', 'ner', 'tagger']

        if pos_filter:
            disable_list.remove('tagger')

        if ner:
            disable_list.remove('ner')

        if return_in_sentence_format:
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
        entities_of_documents = []

        chunked_texts, chunk_list = Preprocesser.chunk_text(texts, 5000)
        # print(texts)
        # print(chunked_texts)
        # fixme MemoryError: Unable to allocate 2.06 GiB for an array with shape (5755403, 96) and data type float32
        for doc in tqdm(nlp.pipe(chunked_texts, disable=disable_list), desc="Preprocessing", total=len(chunked_texts)):
            if ner:
                ents = {e.text: e.label_ for e in doc.ents}
            else:
                ents = {}

            entities_of_documents.append(ents)

            if return_in_sentence_format:
                all_sentences = []
                for sent in doc.sents:
                    sentence_tokens = [token_representation(token)
                                       for token in sent if filter_condition(token)
                                       ]
                    preprocessed_texts.append(sentence_tokens)
                    all_sentences.extend(sentence_tokens)

                preprocessed_documents.append(all_sentences)
            else:
                preprocessed_texts.append(
                    [token_representation(token)
                     for token in doc if filter_condition(token)
                     ]
                )

        # print('texts1', preprocessed_texts)
        if not return_in_sentence_format:
            preoprocessed_unchunked = Preprocesser.merge_chunks(preprocessed_texts, chunk_list)
        else:
            preoprocessed_unchunked = preprocessed_texts
        # print('texts2', preoprocessed_unchunked)
        # todo: check if works with return in sentence format
        # print('documents1', preprocessed_documents)
        documents_unchunked = Preprocesser.merge_chunks(preprocessed_documents, chunk_list)
        # print('documents2', documents_unchunked)
        # print('entities')
        entities_unchunked = Preprocesser.merge_chunks(entities_of_documents, chunk_list)
        # # print(len(preprocessed_texts), len(preoprocessed_unchunked))
        # # print(len(entities_of_documents), len(entities_unchunked))
        # # print(len(preprocessed_documents), len(documents_unchunked))
        # print(entities_unchunked)
        # return preprocessed_texts, entities_of_documents, preprocessed_documents
        return preoprocessed_unchunked, entities_unchunked, documents_unchunked

    @staticmethod
    def filter(corpus: Corpus,
               pos: list = None,
               remove_stopwords: bool = False,
               remove_punctuation: bool = True,
               remove_ne: bool = False,
               masking: bool = False):

        def filter_condition(token: Token):
            return (not remove_stopwords or not token.stop) \
                   and (not remove_punctuation or token.alpha) \
                   and (not pos or token.pos in pos) \
                   and (not remove_ne or not token.ne)

        def mask(token: Token):
            if not filter_condition(token):
                token.text = "del"
                token.lemma = "del"
            return token

        for doc_id, document in tqdm(corpus.documents.items(), total=len(corpus.documents), desc="Filtering corpus"):
            if not masking:
                new_sents = [Sentence([token for token in sentence.tokens if filter_condition(token)])
                              for sentence in document.sentences]
            else:
                new_sents = [Sentence([mask(token) for token in sentence.tokens])
                              for sentence in document.sentences]

            document.set_sentences(new_sents)
        return corpus


