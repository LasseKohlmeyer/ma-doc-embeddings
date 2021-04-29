import json
import os
import random
import re
from collections import defaultdict
from typing import List, Union, Dict
import numpy as np
import spacy
from tqdm import tqdm

from corpus_structure import Language


class ConfigLoader:
    @staticmethod
    def get_config(relative_path=""):

        path = os.path.join(relative_path, "../configs", "config.json")
        if os.path.exists(path):
            with open(path) as json_file:
                return json.load(json_file)

        path = os.path.join(relative_path, "default.config.json")
        if os.path.exists(path):
            path = os.path.join(relative_path, "../configs", "default.config.json")
            with open(path) as json_file:
                return json.load(json_file)

        raise Exception("config file missing!")


config = ConfigLoader.get_config()


class Document:
    __slots__ = 'doc_id', 'text', 'lemma', 'pos', 'ne', 'punctuation', 'alpha', 'stop', 'absolute_positions', \
                'title', 'language', 'authors', 'date', 'genres', 'sentences', \
                'absolute_positions', 'file_path'

    def __init__(self, doc_id: str, file_content: str,
                 title: str, language: Language, authors: str = None, date: str = None, genres: str = None,
                 file_path: str = None,
                 disable_sentences: bool = False):
        self.doc_id = doc_id
        self.title = title
        self.language = language
        self.authors = authors
        self.date = date
        self.genres = genres
        self.file_path = file_path
        self.text, self.lemma, self.pos, self.ne, self.punctuation, self.alpha, self.stop = \
            None, None, None, None, None, None, None

        if not disable_sentences:
            self.build_doc(file_content)

        self.absolute_positions = None

    def build_doc(self, content):
        def parse_sentence(sentence: str):
            return np.array([token.split('\t') for token in sentence.split('\n') if token != ""])

        sentences = np.array([parse_sentence(sentence) for sentence in content.split('<SENT>')
                              if len(sentence) > 0 and sentence != "\n"],
                             dtype="object")

        self.text, self.lemma, self.pos, self.ne, self.punctuation, self.alpha, self.stop = [], [], [], [], [], [], []

        for sentence in sentences:
            # text, lemma = sentence.transpose()
            # self.text.append(text)
            # self.lemma.append(lemma)

            # print('>>', sentence)
            text, lemma, pos, ne, punctuation, alpha, stop = sentence.transpose()
            self.text.append(text)
            self.lemma.append(lemma)
            self.pos.append(pos)
            self.ne.append(ne)
            self.punctuation.append(punctuation)
            self.alpha.append(alpha)
            self.stop.append(stop)

        self.text = np.array(self.text, dtype="object")
        self.lemma = np.array(self.lemma, dtype="object")
        self.pos = np.array(self.pos, dtype="object")
        self.ne = np.array(self.ne, dtype="object")
        self.punctuation = np.array(self.punctuation, dtype="object")
        self.alpha = np.array(self.alpha, dtype="object")
        self.stop = np.array(self.stop, dtype="object")

    def build_position_indices(self):
        c = 0
        for i, sentence in enumerate(self.text):
            for j, token in enumerate(sentence):
                self.absolute_positions[c] = (i, j)
                c += 1

    def get_token_at_position(self, sentence_position: int, token_position: int = None, attribute: str = "text"):
        if token_position is None:
            if len(self.absolute_positions) == 0:
                self.build_position_indices()
            try:
                sentence_position, token_position = self.absolute_positions[sentence_position]
            except KeyError:
                return None

        if attribute == "text":
            return self.text[sentence_position][token_position]
        elif attribute == "lemma":
            return self.lemma[sentence_position][token_position]
        elif attribute == "pos":
            return self.pos[sentence_position][token_position]
        elif attribute == "ne":
            return self.ne[sentence_position][token_position]
        elif attribute == "punctuation":
            return self.punctuation[sentence_position][token_position]
        elif attribute == "alpha":
            return self.alpha[sentence_position][token_position]
        elif attribute == "stop":
            return self.stop[sentence_position][token_position]
        else:
            raise UserWarning(f"Attribute {attribute} not defined!")

    def get_document_sentences(self, lemma: bool = False, lower: bool = False) -> np.array:
        if lemma:
            sentences = self.lemma
        else:
            sentences = self.text
        if lower:
            sentences = np.char.lower(sentences)

        if len(sentences) == 0:
            raise UserWarning("No sentences set")
        return sentences

    def get_flat_document_tokens(self, lemma: bool = False, lower: bool = False) -> np.array:
        if lemma:
            print('>', self.lemma)
            tokens = np.hstack(self.lemma).astype(str)
        else:
            tokens = np.hstack(self.text).astype(str)
        if lower:
            tokens = np.char.lower(tokens)

        if len(tokens) == 0:
            raise UserWarning("No sentences set")
        return tokens

    def get_vocab(self, lemma: bool = False, lower: bool = False) -> np.array:
        return np.unique(self.get_flat_document_tokens(lemma, lower))

    def __str__(self):
        return f'{self.authors} ({self.date}): {self.title[:50]}'

    __repr__ = __str__

    def __len__(self):
        return len(self.get_flat_document_tokens())

    def get_sentence_nr(self):
        return len(self.text)

    # def json_representation(self):
    #     return vars(self)

    def meta_string_representation(self):
        pattern = re.compile(r'[\W]+', re.UNICODE)
        resu = f'{self.doc_id}_-_{str(self.authors).replace(" ", "_")}_-_' \
               f'{pattern.sub("", str(self.title)).replace(" ", "_")}_-_' \
               f'{self.language}_-_{str(self.genres).replace(" ", "_")}_-_{self.date}'
        resu = resu.replace('"', '')
        return resu

    @staticmethod
    def create_document_from_doc_file(doc_path: str, disable_sentences: bool = False):
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

        with open(doc_path, "r", encoding="utf-8") as file:
            doc = Document(doc_id=doc_id, file_content=file.read(), title=title,
                           language=Language.get_from_str(language),
                           authors=authors, date=date, genres=genres, file_path=doc_path,
                           disable_sentences=disable_sentences)
        return doc


# class Corpus:
#     __slots__ = 'name', 'language', 'document_entities', 'series_dict',
#     'root_corpus_path', 'shared_attributes_dict', \
#                 'reversed_attributes_dict', 'success_dict', 'documents', 'file_dict'
#
#     def __init__(self, source: Union[Dict[Union[str, int], Document], List[Document], str],
#                  name: str = None,
#                  language: Language = None):
#         self.name = name
#         self.language = language
#         self.document_entities = None
#         self.series_dict = None
#         self.root_corpus_path = None
#         self.shared_attributes_dict = None
#         self.reversed_attributes_dict = None
#         self.success_dict = None,
#         self.file_dict = None
#
#         if isinstance(source, str):
#             if source.endswith('.json'):
#                 documents, name, language, document_entities, series_dict = self.load_corpus(path=source)
#                 self.name = name
#                 self.language = language
#                 self.document_entities = document_entities
#                 self.series_dict = series_dict
#             else:
#                 other_corpus = self.fast_load(path=source)
#                 self.name = other_corpus.name
#                 self.language = other_corpus.language
#                 self.document_entities = other_corpus.document_entities
#                 self.series_dict = other_corpus.series_dict
#                 documents = other_corpus.documents
#         else:
#             if name is None or language is None:
#                 raise UserWarning("No name or language set!")
#             documents = source
#
#         if isinstance(documents, dict):
#             self.documents: Dict[str, Document] = documents
#         elif isinstance(documents, list):
#             self.documents: Dict[str, Document] = {document.doc_id: document for document in documents}
#         else:
#             self.documents: Dict[str, Document] = {}
#             raise NotImplementedError("Not supported Document collection!")
#
#         self.file_dict = {doc_id: document.file_path for doc_id, document in self.documents.items()}
#         # self.token_number = sum((len(doc) for doc in self.documents.values()))
#
#     def get_documents(self, as_list=True) -> Union[List[Document], Dict[Union[str, int], Document]]:
#         if as_list:
#             return list(self.documents.values())
#         else:
#             return self.documents
#
#     def get_n_documents_as_corpus(self, n: int) -> "Corpus":
#         documents = self.get_documents(as_list=True)
#         documents = documents[:n]
#         return Corpus(source=documents,
#                       language=self.language,
#                       name=f'{self.name}_top{n}')
#
#     def save_corpus_adv(self, corpus_dir: str):
#         if not os.path.isdir(corpus_dir):
#             os.mkdir(corpus_dir)
#         for doc_id, document in tqdm(self.documents.items(), total=len(self.documents), desc="save corpus",
#                                      disable=False):
#             document.meta_string_representation()
#             doc_path = os.path.join(corpus_dir, f'{document.meta_string_representation()}.txt')
#             doc_sents = []
#             with open(doc_path, 'w', encoding="utf-8") as writer:
#                 # writer.write("Text\tLemma\tPOS\tNE\tPunctuation\tAlphabetic\tStopword")
#                 for sentence in document.sentences:
#                     for token in sentence.tokens:
#                         writer.write(f'{token.get_save_file_representation()}\n')
#                     writer.write("<SENT>\n")
#                     doc_sents.append(sentence)
#             document.sentences = doc_sents
#         if self.root_corpus_path is None:
#             self.root_corpus_path = corpus_dir
#         data = {"name": self.name, "root_corpus_path": self.root_corpus_path,
#                 "language": self.language, "series_dict": self.series_dict,
#                 "success_dict": self.success_dict}
#         print('write meta')
#         with open(os.path.join(corpus_dir, "meta_info.json"), 'w', encoding='utf-8') as f:
#             json.dump(data, f, ensure_ascii=False, indent=1, default=lambda o: o.__dict__)
#
#     @staticmethod
#     def load_corpus_from_dir_format(corpus_dir: str):
#         meta_path = os.path.join(corpus_dir, "meta_info.json")
#         with open(meta_path, 'r', encoding='utf-8') as file:
#             meta_data = json.loads(file.read())
#
#         document_paths = [file_path for file_path in os.listdir(corpus_dir) if file_path.endswith('.txt')][:1000]
#
#         documents = [Document.create_document_from_doc_file(os.path.join(corpus_dir, doc_path),
#         disable_sentences=False)
#                      for doc_path in tqdm(document_paths, desc="load_file", disable=False)]
#
#         corpus = Corpus(source=documents, name=meta_data["name"], language=meta_data["language"])
#
#         corpus.root_corpus_path = meta_data["root_corpus_path"]
#         corpus.set_series_dict(meta_data["series_dict"])
#         if "success_dict" in meta_data.keys():
#             corpus.success_dict = meta_data["success_dict"]
#
#         return corpus
#
#     @staticmethod
#     def fast_load(number_of_subparts=None, size=None, data_set=None, filer_mode=None, fake_real=None, path=None,
#                   load_entities: bool = True):
#         if path is None:
#             corpus_dir = Corpus.build_corpus_dir(number_of_subparts,
#                                                  size,
#                                                  data_set,
#                                                  filer_mode,
#                                                  fake_real)
#             if os.path.exists(corpus_dir):
#                 corpus = Corpus.load_corpus_from_dir_format(corpus_dir)
#             else:
#                 corpus_path = Corpus.build_corpus_file_name(number_of_subparts,
#                                                             size,
#                                                             data_set,
#                                                             filer_mode,
#                                                             fake_real)
#                 corpus = Corpus(corpus_path)
#                 corpus.save_corpus_adv(corpus_dir)
#         else:
#             if os.path.exists(path):
#                 corpus = Corpus.load_corpus_from_dir_format(path)
#             else:
#                 corpus = Corpus(f'{path}.json')
#                 corpus.save_corpus_adv(path)
#
#         # corpus.set_sentences_from_own_gens()
#         if load_entities:
#             corpus.set_document_entities()
#
#         return corpus
#
#     def get_years(self) -> [str]:
#         years = set()
#         for d in self.get_documents(as_list=True):
#             if d.date:
#                 years.add(d.date)
#         return sorted(list(years))
#
#     @staticmethod
#     def build_corpus_file_name(number_of_subparts: Union[int, str], size: Union[int, str],
#                                dataset: str, filter_mode: str, fake_series: str) -> str:
#         sub_path = DataHandler.build_config_str(number_of_subparts, size, dataset, filter_mode,
#                                                 '', fake_series)
#         return os.path.join(config["system_storage"]["corpora"], f'{sub_path}.json')
#
#     @staticmethod
#     def build_corpus_name(number_of_subparts: Union[int, str], size: Union[int, str],
#                           dataset: str, filter_mode: str, fake_series: str) -> str:
#         return DataHandler.build_config_str(number_of_subparts, size, dataset, filter_mode,
#                                             '', fake_series)
#
#     @staticmethod
#     def build_corpus_dir(number_of_subparts: Union[int, str], size: Union[int, str],
#                          dataset: str, filter_mode: str, fake_series: str) -> str:
#         sub_path = DataHandler.build_config_str(number_of_subparts, size, dataset, filter_mode,
#                                                 '', fake_series)
#         return os.path.join(config["system_storage"]["corpora"], sub_path)
#
#     @staticmethod
#     def load_corpus(path: str):
#         with open(path, 'r', encoding='utf-8') as file:
#             data = json.loads(file.read())
#
#         # doc_sents = []
#         # for doc_id, doc in data["documents"].items():
#         #     sentences = [Sentence([Token(text=token["text"],
#         #                         lemma=token["lemma"],
#         #                         pos=token["pos"],
#         #                         ne=token["ne"],
#         #                         punctuation=token["punctuation"],
#         #                         alpha=token["alpha"],
#         #                         stop=token["stop"])
#         #                            for token in sentence["tokens"]])
#         #                  for sentence in doc["sentences"]]
#         #     doc_sents.append(sentences)
#         # print(doc_sents)
#
#         documents = {doc_id: Document(doc_id=doc["doc_id"],
#                                       text=doc["text"],
#                                       title=doc["title"],
#                                       language=Language.get_from_str(doc["language"]),
#                                       authors=doc["authors"],
#                                       date=doc["date"],
#                                       genres=doc["genres"],
#                                       sentences=[Sentence([Token(text=token["text"],
#                                                                  lemma=token["lemma"],
#                                                                  pos=token["pos"],
#                                                                  ne=token["ne"],
#                                                                  punctuation=token["punctuation"],
#                                                                  alpha=token["alpha"],
#                                                                  stop=token["stop"])
#                                                            for token in sentence["tokens"]])
#                                                  for sentence in doc["sentences"]])
#                      for doc_id, doc in data["documents"].items()}
#         language = data["language"]
#         name = data["name"]
#
#         document_entities = {doc_id: defaultdict(list, {en: [Token(text=token["text"],
#                                                                    lemma=token["lemma"],
#                                                                    pos=token["pos"],
#                                                                    ne=token["ne"],
#                                                                    punctuation=token["punctuation"],
#                                                                    alpha=token["alpha"],
#                                                                    stop=token["stop"]) for token in tokens]
#                                                         for en, tokens in doc_data.items()})
#                              for doc_id, doc_data in data["document_entities"].items()}
#
#         if data["series_dict"] is None:
#             series_dict = defaultdict(list)
#         else:
#             series_dict = defaultdict(list, data["series_dict"])
#
#         return documents, name, language, document_entities, series_dict
#
#     def get_corpus_vocab(self):
#         vocab = set()
#         for document in self.documents.values():
#             vocab.update(document.get_corpus_vocab())
#         if 'del' in vocab:
#             vocab.remove('del')
#         return vocab
#
#     def sample(self, number_documents=100, seed=None):
#         if len(self.documents) < number_documents:
#             return self
#         if seed:
#             random.seed(seed)
#         result = Corpus(source=random.sample(self.get_documents(), k=number_documents),
#                         language=self.language,
#                         name=f'{self.name}_{number_documents}_sample')
#         return result
#
#     def id2desc(self, index: Union[str, int]):
#         if index.endswith('_sum'):
#             index = index.replace('_sum', '')
#         elif index.endswith('_time'):
#             index = index.replace('_time', '')
#         elif index.endswith('_loc'):
#             index = index.replace('_loc', '')
#         return str(self.documents[index])
#
#     def give_spacy_lan_model(self):
#         if self.language == Language.EN:
#             return spacy.load("en_core_web_sm")
#         else:
#             return spacy.load("de_core_news_sm")
#
#     def set_document_entities(self):
#         # ents = {e.text: e.label_ for e in doc.ents}
#         # entities_of_documents.append(ents)
#         # todo fixme
#         entities_dict = {}
#         for doc_id, doc in self.documents.items():
#             doc: Document
#             doc_entities = defaultdict(list)
#             for sent in doc.sentences:
#                 for token in sent.tokens:
#                     if token.ne:
#                         # print(token.ne, token.text)
#                         doc_entities[token.ne].append(token)
#             entities_dict[doc_id] = doc_entities
#         # print(entities_dict)
#         self.document_entities = entities_dict
#
#     def update_time_entities(self, update_dict: Dict[str, List[str]]):
#         def find_sub_list(sub_list, main_list):
#             results = []
#             sll = len(sub_list)
#             for ind in (j for j, e in enumerate(main_list) if e == sub_list[0]):
#                 if main_list[ind:ind + sll] == sub_list:
#                     results.append((ind, ind + sll - 1))
#
#             return results
#
#         for doc_id, time_ents in tqdm(update_dict.items(), total=len(update_dict)):
#             time_entities = set(time_ents)
#             if doc_id in self.documents.keys():
#                 token_reprs = [token.representation() for sentence in self.documents[doc_id].sentences
#                                for token in sentence.tokens]
#                 for time_entity in time_entities:
#                     tm = time_entity.split(' ')
#                     positions = find_sub_list(tm, token_reprs)
#                     for position in positions:
#                         start, end = position
#                         for i in range(start, end+1):
#                             self.documents[doc_id].get_token_at_doc_position(i).ne = "TIME"
#
#     def set_series_dict(self, series_dict: Dict[str, List[str]]):
#         self.series_dict = series_dict
#
#     def get_document_entities_representation(self, lemma=False, lower=False):
#         return {doc_id: {entity_type: [token.representation(lemma=lemma, lower=lower) for token in tokens]
#                          for entity_type, tokens in entities.items()}
#                 for doc_id, entities in self.document_entities.items()}
#
#     # def set_root_path(self, root_path: str):
#     #     self.root_corpus_path = root_path
#     def get_flat_documents(self, lemma: bool = False, lower: bool = False, as_sentence: bool = True):
#         if as_sentence:
#             documents = [' '.join(document.get_flat_document_tokens(lemma, lower))
#                          for doc_id, document in self.documents.items()]
#         else:
#             documents = [document.get_flat_document_tokens(lemma, lower)
#             for doc_id, document in self.documents.items()]
#         if len(documents) == 0:
#             raise UserWarning("No sentences set")
#
#         return documents
#
#     def get_flat_document_tokens(self, lemma: bool = False, lower: bool = False, as_dict: bool = False) -> \
#             Union[List[str],  Dict[str, List[str]]]:
#         # for doc_id, document in self.documents.items():
#         #     print(document.sentences[1].tokens)
#         if as_dict:
#             tokens = {doc_id: document.get_flat_document_tokens(lemma, lower)
#                       for doc_id, document in self.documents.items()}
#         else:
#             tokens = [document.get_flat_document_tokens(lemma, lower)
#                       for doc_id, document in self.documents.items()]
#         if len(tokens) == 0:
#             raise UserWarning("No sentences set")
#         return tokens
#
#     def get_tokens_from_file(self, doc_id):
#         return Document.create_document_from_doc_file(self.file_dict[doc_id])
#
#     def get_flat_and_filtered_document_tokens(self, lemma: bool = False, lower: bool = False, pos: list = None,
#                                               focus_stopwords: bool = False,
#                                               focus_punctuation: bool = False,
#                                               focus_ne: bool = False,
#                                               masking: bool = False,
#                                               revert: bool = False):
#         # todo fix
#         def filter_condition(token: Token):
#             if revert:
#                 return (not focus_stopwords or not token.stop) \
#                        and (not focus_punctuation or not token.alpha) \
#                        and (not pos or token.pos not in pos) \
#                        and (not focus_ne or not token.ne)
#             else:
#                 return (not focus_stopwords or token.stop) \
#                        and (not focus_punctuation or token.alpha) \
#                        and (not pos or token.pos in pos) \
#                        and (not focus_ne or token.ne)
#
#         def mask(input_token: Token):
#             output_token = Token(text=input_token.text,
#                                  lemma=input_token.lemma,
#                                  pos=input_token.pos,
#                                  ne=input_token.ne,
#                                  punctuation=input_token.punctuation,
#                                  alpha=input_token.alpha,
#                                  stop=input_token.stop)
#             if not filter_condition(output_token):
#                 output_token.text = "del"
#                 output_token.lemma = "del"
#             return output_token
#
#         # for doc_id, document in self.documents.items():
#         #     print(document.sentences[1].tokens)
#         if not masking:
#             tokens = [[token.representation(lemma, lower)
#                        for sentence in document.sentences
#                        for token in sentence.tokens
#                        if filter_condition(token)]
#                       for doc_id, document in self.documents.items()]
#         else:
#             tokens = [[mask(token).representation(lemma, lower)
#                        for sentence in document.sentences
#                        for token in sentence.tokens]
#                       for doc_id, document in self.documents.items()]
#
#         if len(tokens) == 0:
#             raise UserWarning("No sentences set")
#
#         return tokens
#
#     def get_flat_and_random_document_tokens(self,  prop_to_keep: float, seed: int,
#                                             lemma: bool = False, lower: bool = False,
#                                             masking: bool = False):
#         # todo fix
#         def filter_condition(token: Token):
#             random_number = random.randint(1, 1000)
#             return random_number <= 1000*prop_to_keep
#
#         def mask(input_token: Token):
#             output_token = Token(text=input_token.text,
#                                  lemma=input_token.lemma,
#                                  pos=input_token.pos,
#                                  ne=input_token.ne,
#                                  punctuation=input_token.punctuation,
#                                  alpha=input_token.alpha,
#                                  stop=input_token.stop)
#             if not filter_condition(output_token):
#                 output_token.text = "del"
#                 output_token.lemma = "del"
#             return output_token
#         random.seed(seed)
#         if not masking:
#             tokens = [[token.representation(lemma, lower)
#                        for sentence in document.sentences
#                        for token in sentence.tokens
#                        if filter_condition(token)]
#                       for doc_id, document in self.documents.items()]
#         else:
#             tokens = [[mask(token).representation(lemma, lower)
#                        for sentence in document.sentences
#                        for token in sentence.tokens]
#                       for doc_id, document in self.documents.items()]
#
#         if len(tokens) == 0:
#             raise UserWarning("No sentences set")
#
#         return tokens
#
#     def get_flat_corpus_sentences(self, lemma: bool = False, lower: bool = False):
#         sentences = []
#         for document in self.documents.values():
#             sentences.extend(document.get_document_sentences(lemma, lower))
#
#         if len(sentences) == 0:
#             raise UserWarning("No sentences set")
#         return sentences
#
#     def calculate_documents_with_shared_attributes(self):
#         same_author_dict = defaultdict(list)
#         same_year_dict = defaultdict(list)
#         same_genre_dict = defaultdict(list)
#
#         for doc_id, document in self.documents.items():
#             same_author_dict[document.authors].append(doc_id)
#             same_year_dict[document.date].append(doc_id)
#             same_genre_dict[document.genres].append(doc_id)
#         self.shared_attributes_dict = {"same_author": same_author_dict, "same_year": same_year_dict,
#                                        "same_genre_dict": same_genre_dict}
#
#         self.reversed_attributes_dict = {category: Utils.revert_dictionaried_list(category_dict)
#                                          for category, category_dict in self.shared_attributes_dict.items()}
#         # print(self.shared_attributes_dict["same_author"])
#
#     def get_other_doc_ids_by_same_author(self, doc_id):
#         # same_author_docs = [document for document in self.documents.values()
#         #                     if document.doc_id != doc_id and document.authors == self.documents[doc_id].authors]
#         # return same_author_docs
#         if self.shared_attributes_dict is None:
#             self.calculate_documents_with_shared_attributes()
#         other_ids = self.shared_attributes_dict["same_author"][self.documents[doc_id].authors]
#         return other_ids
#
#     def get_other_doc_ids_by_same_genres(self, doc_id):
#         if self.shared_attributes_dict is None:
#             self.calculate_documents_with_shared_attributes()
#         other_ids = self.shared_attributes_dict["same_genres"][self.documents[doc_id].genres].remove(doc_id)
#         return other_ids
#
#     def get_other_doc_ids_by_same_year(self, doc_id):
#         if self.shared_attributes_dict is None:
#             self.calculate_documents_with_shared_attributes()
#         other_ids = self.shared_attributes_dict["same_year"][self.documents[doc_id].date].remove(doc_id)
#         return other_ids
#
#     def get_windowed_aspects(self, aspect_dict: Dict[str, List[str]], window_size: int = 10):
#         context_aspect_dict = {}
#         for aspect, aspect_docs in aspect_dict.items():
#             windowed_docs = []
#             for (doc_id, document), aspect_doc in zip(self.documents.items(), aspect_docs):
#                 windowed_sentence = []
#
#                 for i, sentence in enumerate(document.sentences):
#                     token_id_matches = []
#
#                     for j, token in enumerate(sentence.tokens):
#                         if token.representation() == aspect_doc[i][j]:
#                             token_id_matches.append(j)
#                     for matched_id in token_id_matches:
#                         min_id = matched_id - window_size
#                         max_id = matched_id + window_size
#                         if min_id < 0:
#                             min_id = 0
#                         if max_id > len(sentence.tokens) - 1:
#                             max_id = len(sentence.tokens) - 1
#
#                         windowed_sentence.extend(sentence.tokens[min_id, max_id])
#                 windowed_docs.append(windowed_sentence)
#
#             context_aspect_dict[aspect] = windowed_docs
#         return context_aspect_dict
#
#     def fake_series(self, number_of_sub_parts=2) -> Tuple["Corpus", Dict[str, List[str]]]:
#         fake_series_corpus = []
#         fake_series_dict = defaultdict(list)
#
#         for doc_id, document in self.documents.items():
#             # print(len(document.sentences), number_of_sub_parts)
#             if len(document.sentences) < number_of_sub_parts:
#                 raise UserWarning("Nr of document sentences too small!")
#             sentence_counter = 0
#             # avg_doc_length = math.ceil(len(document.sentences) / number_of_sub_parts)
#             avg_doc_length = len(document.sentences) // number_of_sub_parts
#             # print(doc_id, len(document.sentences))
#             for i in range(0, number_of_sub_parts):
#                 series_doc_id = f'{doc_id}_{i}'
#                 fake_series_dict[doc_id].append(series_doc_id)
#                 fake_series_doc = Document(doc_id=series_doc_id,
#                                            text=document.text,
#                                            title=f'{document.title} {i}',
#                                            language=document.language,
#                                            authors=document.authors,
#                                            date=document.date,
#                                            genres=document.genres)
#                 if i + 1 == number_of_sub_parts:
#                     end = None
#                 else:
#                     end = (i + 1) * avg_doc_length
#                 sub_sentences = document.sentences[i * avg_doc_length:end]
#                 fake_series_doc.set_sentences(sub_sentences)
#                 fake_series_doc.reset_text_based_on_sentences()
#                 # if len(fake_series_doc.sentences) == 0:
#                 #     print(document.date, document.doc_id, document.title, fake_series_doc.text, document.text)
#                 #     print(sentence_counter, len(document.sentences), avg_doc_length)
#                 fake_series_corpus.append(fake_series_doc)
#                 sentence_counter += len(fake_series_doc.sentences)
#
#                 assert len(fake_series_doc.sentences) > 0
#             assert sentence_counter == len(document.sentences)
#         fake_series_corpus = Corpus(fake_series_corpus, name=f'{self.name}_fake', language=self.language)
#         fake_series_corpus.set_document_entities()
#         fake_series_corpus.set_series_dict(fake_series_dict)
#         # for doc_id, doc in fake_series_corpus.documents.items():
#         #     print(doc_id, doc.text)
#         # print(fake_series_corpus)
#         return fake_series_corpus, fake_series_dict
#
#     def get_common_words_relaxed(self, series_dictionary: Dict[str, List[str]]) -> Dict[str, Set[str]]:
#         common_words = CommonWords.relaxed(series_dictionary, self.get_flat_document_tokens(as_dict=True))
#         for series_id, doc_ids in series_dictionary.items():
#             for doc_id in doc_ids:
#                 common_words[doc_id] = common_words[series_id]
#         return common_words
#
#     def get_common_words_strict(self, series_dictionary: Dict[str, List[str]]) -> Dict[str, Set[str]]:
#         common_words = CommonWords.strict(series_dictionary, self.get_flat_document_tokens(as_dict=True))
#         for series_id, doc_ids in series_dictionary.items():
#             for doc_id in doc_ids:
#                 common_words[doc_id] = common_words[series_id]
#         return common_words
#
#     def get_common_words_relaxed_gen_words(self, series_dictionary: Dict[str, List[str]]) -> Dict[str, Set[str]]:
#         common_words = CommonWords.relaxed_general_words_sensitive(series_dictionary,
#                                                                    self.get_flat_document_tokens(as_dict=True))
#         for series_id, doc_ids in series_dictionary.items():
#             for doc_id in doc_ids:
#                 common_words[doc_id] = common_words[series_id]
#         return common_words
#
#     def get_common_words_strict_gen_words(self, series_dictionary: Dict[str, List[str]]) -> Dict[str, Set[str]]:
#         common_words = CommonWords.strict_general_words_sensitive(series_dictionary,
#                                                                   self.get_flat_document_tokens(as_dict=True))
#         for series_id, doc_ids in series_dictionary.items():
#             for doc_id in doc_ids:
#                 common_words[doc_id] = common_words[series_id]
#         return common_words
#
#     def get_global_common_words(self) -> Set[str]:
#         common_words = CommonWords.global_too_specific_words_doc_frequency(self.get_flat_document_tokens(as_dict=True),
#                                                                            percentage_share=0.25)
#         return common_words
#
#     # def get_common_words(self, series_dictionary: Dict[str, List[str]]) -> Dict[str, Set[str]]:
#     #     common_words = defaultdict(set)
#     #     for series_id, doc_ids in series_dictionary.items():
#     #         series_words = []
#     #         for doc_id in doc_ids:
#     #             series_words.append(set(self.documents[doc_id].get_flat_document_tokens(lemma=True, lower=True)))
#     #
#     #         for token_set_a in series_words:
#     #             for token_set_b in series_words:
#     #                 if token_set_a != token_set_b:
#     #                     common_words[series_id].update(token_set_a.intersection(token_set_b))
#     #         # common_words[series_id] = set.intersection(*series_words)
#     #         for doc_id in doc_ids:
#     #             common_words[doc_id] = common_words[series_id]
#     #     return common_words
#
#     def common_words_corpus_filtered(self, common_words: Union[Set[str], Dict[str, Set[str]]], masking: bool):
#         def filter_condition(token: Token, document_id: str, common_ws: Union[Set[str], Dict[str, Set[str]]]):
#             # print(token.representation(lemma=True, lower=True) not in common_words[document_id],
#             #       token.representation(lemma=True, lower=True),
#             #       common_words[document_id])
#             if isinstance(common_words, set):
#                 return token.representation(lemma=False, lower=False) not in common_ws
#             else:
#                 return token.representation(lemma=False, lower=False) not in common_ws[document_id]
#
#         def mask(token: Token, document_id: str, common_ws: Set[str]):
#             if not filter_condition(token, document_id, common_ws):
#                 token.text = "del"
#                 token.lemma = "del"
#             return token
#         for doc_id, document in tqdm(self.documents.items(), total=len(self.documents), desc="Filtering corpus",
#                                      disable=True):
#             if not masking:
#                 new_sents_gen = (Sentence([token for token in sentence.tokens
#                                            if filter_condition(token, doc_id, common_words)])
#                                  for sentence in document.sentences)
#                 new_sents = []
#                 for sent in new_sents_gen:
#                     if len(sent.tokens) == 0:
#                         sent.tokens.append(Token.empty_token())
#                     new_sents.append(sent)
#                 # for new_s in new_sents:
#                 #     print(new_s.representation())
#             else:
#                 new_sents = [Sentence([mask(token, doc_id, common_words) for token in sentence.tokens])
#                              for sentence in document.sentences]
#                 # for new_s in new_sents:
#                 #     print(new_s.representation())
#             document.set_sentences(new_sents)
#
#         #     documents[doc_id] = Document(doc_id=doc_id,
#         #                                  text=document.text,
#         #                                  title=document.title,
#         #                                  language=document.language,
#         #                                  authors=document.authors,
#         #                                  date=document.date,
#         #                                  genres=document.genres,
#         #                                  sentences=new_sents)
#         #
#         # common_words_corpus = Corpus(documents, self.name, self.language)
#         # common_words_corpus.set_series_dict(self.series_dict)
#         # common_words_corpus.set_document_entities()
#         # common_words_corpus.file_dict = self.file_dict
#
#     def common_words_corpus_copy(self, common_words: Union[Set[str], Dict[str, Set[str]]], masking: bool):
#         def filter_condition(token: Token, document_id: str, common_ws: Union[Set[str], Dict[str, Set[str]]]):
#             # print(token.representation(lemma=True, lower=True) not in common_words[document_id],
#             #       token.representation(lemma=True, lower=True),
#             #       common_words[document_id])
#             if isinstance(common_words, set):
#                 return token.representation(lemma=False, lower=False) not in common_ws
#             else:
#                 return token.representation(lemma=False, lower=False) not in common_ws[document_id]
#
#         def mask(token: Token, document_id: str, common_ws: Set[str]):
#             if not filter_condition(token, document_id, common_ws):
#                 token.text = "del"
#                 token.lemma = "del"
#             return token
#
#         documents = {}
#         for doc_id, document in tqdm(self.documents.items(), total=len(self.documents), desc="Filtering corpus",
#                                      disable=True):
#             if not masking:
#                 new_sents_gen = (Sentence([token for token in sentence.tokens
#                                            if filter_condition(token, doc_id, common_words)])
#                                  for sentence in document.sentences)
#                 new_sents = []
#                 for sent in new_sents_gen:
#                     if len(sent.tokens) == 0:
#                         sent.tokens.append(Token.empty_token())
#                     new_sents.append(sent)
#                 # for new_s in new_sents:
#                 #     print(new_s.representation())
#             else:
#                 new_sents = [Sentence([mask(token, doc_id, common_words) for token in sentence.tokens])
#                              for sentence in document.sentences]
#                 # for new_s in new_sents:
#                 #     print(new_s.representation())
#             # document.set_sentences(new_sents)
#
#             documents[doc_id] = Document(doc_id=doc_id,
#                                          text=document.text,
#                                          title=document.title,
#                                          language=document.language,
#                                          authors=document.authors,
#                                          date=document.date,
#                                          genres=document.genres,
#                                          sentences=new_sents)
#
#         common_words_corpus = Corpus(documents, self.name, self.language)
#         common_words_corpus.set_series_dict(self.series_dict)
#         common_words_corpus.set_document_entities()
#         common_words_corpus.file_dict = self.file_dict
#         return common_words_corpus
#
#     def common_words_corpus_copy_improved(self, common_words: Union[Set[str], Dict[str, Set[str]]], masking: bool):
#         def filter_condition(token: Token, document_id: str, common_ws: Union[Set[str], Dict[str, Set[str]]]):
#             # print(token.representation(lemma=True, lower=True) not in common_words[document_id],
#             #       token.representation(lemma=True, lower=True),
#             #       common_words[document_id])
#             if isinstance(common_words, set):
#                 return token.representation(lemma=False, lower=False) not in common_ws
#             else:
#                 return token.representation(lemma=False, lower=False) not in common_ws[document_id]
#
#         def mask(token: Token, document_id: str, common_ws: Set[str]):
#             if not filter_condition(token, document_id, common_ws):
#                 token.text = "del"
#                 token.lemma = "del"
#             return token
#
#         documents = {}
#         for doc_id, _ in tqdm(self.documents.items(), total=len(self.documents), desc="Filtering corpus",
#                               disable=True):
#             document = Document.create_document_from_doc_file(doc_path=self.file_dict[doc_id])
#             if not masking:
#                 new_sents_gen = (Sentence([token for token in sentence.tokens
#                                            if filter_condition(token, doc_id, common_words)])
#                                  for sentence in document.sentences)
#                 new_sents = []
#                 for sent in new_sents_gen:
#                     if len(sent.tokens) == 0:
#                         sent.tokens.append(Token.empty_token())
#                     new_sents.append(sent)
#                 # for new_s in new_sents:
#                 #     print(new_s.representation())
#             else:
#                 new_sents = [Sentence([mask(token, doc_id, common_words) for token in sentence.tokens])
#                              for sentence in document.sentences]
#                 # for new_s in new_sents:
#                 #     print(new_s.representation())
#             # document.set_sentences(new_sents)
#
#             documents[doc_id] = Document(doc_id=doc_id,
#                                          text=document.text,
#                                          title=document.title,
#                                          language=document.language,
#                                          authors=document.authors,
#                                          date=document.date,
#                                          genres=document.genres,
#                                          sentences=new_sents)
#
#         common_words_corpus = Corpus(documents, self.name, self.language)
#         common_words_corpus.set_series_dict(self.series_dict)
#         common_words_corpus.set_document_entities()
#         common_words_corpus.file_dict = self.file_dict
#         return common_words_corpus
#
#     def filter(self, mode: str, masking: bool = False, common_words: Dict[str, Set[str]] = None):
#         def filter_condition(token: Token, document_id: str):
#             # print(token.representation(lemma=True, lower=True) not in common_words[document_id],
#             #       token.representation(lemma=True, lower=True),
#             #       common_words[document_id])
#             return token.representation(lemma=False, lower=False) not in common_words[document_id]
#
#         def mask(token: Token, document_id: str):
#             if not filter_condition(token, document_id):
#                 token.text = "del"
#                 token.lemma = "del"
#             return token
#
#         if mode.lower() == "no_filter" or mode.lower() == "nf":
#             pass
#             # return self
#         elif mode.lower() == "common_words" or mode.lower() == "cw":
#             # print('>>', common_words["bs_0"])
#             for doc_id, document in tqdm(self.documents.items(), total=len(self.documents), desc="Filtering corpus",
#                                          disable=True):
#                 if not masking:
#                     new_sents = [Sentence([token for token in sentence.tokens if filter_condition(token, doc_id)])
#                                  for sentence in document.sentences]
#                     for sent in new_sents:
#                         if len(sent.tokens) == 0:
#                             sent.tokens.append(Token.empty_token())
#                     # for new_s in new_sents:
#                     #     print(new_s.representation())
#                 else:
#                     new_sents = [Sentence([mask(token, doc_id) for token in sentence.tokens])
#                                  for sentence in document.sentences]
#                     # for new_s in new_sents:
#                     #     print(new_s.representation())
#                 document.set_sentences(new_sents)
#                 # return self
#         else:
#             pos = None
#             remove_stopwords = False
#             remove_punctuation = False
#             remove_ne = False
#             revert = False
#
#             if mode.lower() == "named_entities" or mode.lower() == "ne" or mode.lower() == "named_entity":
#                 remove_ne = True
#                 pos = ["PROPN"]
#             elif mode.lower() == "nouns" or mode.lower() == "n" or mode.lower() == "noun":
#                 pos = ["NOUN", "PROPN"]
#                 remove_ne = True
#             elif mode.lower() == "verbs" or mode.lower() == "v" or mode.lower() == "verb":
#                 pos = ["VERB", "ADV"]
#             elif mode.lower() == "adjectives" or mode.lower() == "a" or mode.lower() == "adj" \
#                     or mode.lower() == "adjective":
#                 pos = ["ADJ"]
#             elif mode.lower() == "avn" or mode.lower() == "anv" or mode.lower() == "nav" or mode.lower() == "nva" \
#                     or mode.lower() == "van" or mode.lower() == "vna":
#                 remove_ne = True
#                 pos = ["NOUN", "PROPN", "ADJ", "VERB", "ADV"]
#             elif mode.lower() == "stopwords" or mode.lower() == "stop_words" \
#                     or mode.lower() == "stopword" or mode.lower() == "stop_word" \
#                     or mode.lower() == "stop" or mode.lower() == "sw":
#                 remove_stopwords = True
#             elif mode.lower() == "punctuation" or mode.lower() == "punct" \
#                     or mode.lower() == "." or mode.lower() == "pun" \
#                     or mode.lower() == "punc" or mode.lower() == "zeichen":
#                 remove_punctuation = True
#             else:
#                 raise UserWarning("Not supported mode")
#             Preprocesser.filter(self,
#                                 pos=pos,
#                                 remove_stopwords=remove_stopwords,
#                                 remove_punctuation=remove_punctuation,
#                                 remove_ne=remove_ne,
#                                 masking=masking,
#                                 revert=revert)
#
#     def filter_on_copy(self, mode: str, masking: bool = False) -> "Corpus":
#         def filter_condition(token: Token, document_id: str, common_ws: Dict[str, Set[str]]):
#             # print(token.representation(lemma=True, lower=True) not in common_words[document_id],
#             #       token.representation(lemma=True, lower=True),
#             #       common_words[document_id])
#             return token.representation(lemma=False, lower=False) not in common_ws[document_id]
#
#         def mask(token: Token, document_id: str, common_ws: Dict[str, Set[str]]):
#             if not filter_condition(token, document_id, common_ws):
#                 token.text = "del"
#                 token.lemma = "del"
#             return token
#
#         if mode.lower() == "no_filter" or mode.lower() == "nf":
#             return self
#         elif mode.lower() == "common_words_relaxed" or mode.lower() == "cw_rel":
#             common_words_of_task = self.get_common_words_relaxed(self.series_dict)
#             corpus = self.common_words_corpus_copy(common_words_of_task, masking)
#             return corpus
#         elif mode.lower() == "common_words_strict" or mode.lower() == "cw_str":
#             common_words_of_task = self.get_common_words_strict(self.series_dict)
#             corpus = self.common_words_corpus_copy(common_words_of_task, masking)
#             return corpus
#         elif mode.lower() == "common_words_relaxed_general_words_sensitive" or mode.lower() == "cw_rel_gw":
#             common_words_of_task = self.get_common_words_relaxed_gen_words(self.series_dict)
#             corpus = self.common_words_corpus_copy(common_words_of_task, masking)
#             return corpus
#         elif mode.lower() == "common_words_strict_general_words_sensitive" or mode.lower() == "cw_str_gw":
#             common_words_of_task = self.get_common_words_strict_gen_words(self.series_dict)
#             corpus = self.common_words_corpus_copy(common_words_of_task, masking)
#             return corpus
#         elif mode.lower() == "common_words_doc_freq" or mode.lower() == "cw_df":
#             common_words = self.get_global_common_words()
#             corpus = self.common_words_corpus_copy(common_words, masking)
#             return corpus
#
#         else:
#             pos = None
#             remove_stopwords = False
#             remove_punctuation = False
#             remove_ne = False
#             revert = False
#
#             if mode.lower() == "named_entities" or mode.lower() == "ne" or mode.lower() == "named_entity":
#                 remove_ne = True
#                 pos = ["PROPN"]
#             elif mode.lower() == "nouns" or mode.lower() == "n" or mode.lower() == "noun":
#                 pos = ["NOUN", "PROPN"]
#                 remove_ne = True
#             elif mode.lower() == "verbs" or mode.lower() == "v" or mode.lower() == "verb":
#                 pos = ["VERB", "ADV"]
#             elif mode.lower() == "adjectives" or mode.lower() == "a" or mode.lower() == "adj" \
#                     or mode.lower() == "adjective":
#                 pos = ["ADJ"]
#             elif mode.lower() == "avn" or mode.lower() == "anv" or mode.lower() == "nav" or mode.lower() == "nva" \
#                     or mode.lower() == "van" or mode.lower() == "vna":
#                 remove_ne = True
#                 pos = ["NOUN", "PROPN", "ADJ", "VERB", "ADV"]
#             elif mode.lower() == "stopwords" or mode.lower() == "stop_words" \
#                     or mode.lower() == "stopword" or mode.lower() == "stop_word" \
#                     or mode.lower() == "stop" or mode.lower() == "sw":
#                 remove_stopwords = True
#             elif mode.lower() == "punctuation" or mode.lower() == "punct" \
#                     or mode.lower() == "." or mode.lower() == "pun" \
#                     or mode.lower() == "punc" or mode.lower() == "zeichen":
#                 remove_punctuation = True
#             else:
#                 raise UserWarning(f"Not supported mode: {mode}")
#             return Preprocesser.filter_on_copy(self,
#                                                pos=pos,
#                                                remove_stopwords=remove_stopwords,
#                                                remove_punctuation=remove_punctuation,
#                                                remove_ne=remove_ne,
#                                                masking=masking,
#                                                revert=revert)
#
#     def __iter__(self):
#         return self.documents.values().__iter__()
#
#     def __len__(self):
#         return len(self.documents)
#
#     def token_number(self):
#         return self.token_number  # sum((len(doc) for doc in self.documents.values()))
#
#     def __str__(self):
#         return f'docs={len(self)}, lan={self.language}, name={self.name}'
#
#     def __getitem__(self, key):
#         if isinstance(key, slice):
#             # do your handling for a slice object:
#             # print(key.start, key.stop, key.step)
#             return Corpus(source=list(self.documents.values())[key.start: key.stop: key.step],
#                           name=f'{self.name}_{key.start}_{key.stop}_{key.step}',
#                           language=self.language)
#         elif isinstance(key, int):
#             # print(key)
#             return list(self.documents.values())[key]
#         else:
#             # Do your handling for a plain index
#             # print(key)
#             return self.documents[key]
#
#     __repr__ = __str__
#
#
# # d = Document(doc_id="d1", file_content="hallo\tBEGR\ndas\tART\nist\tVERB\nmeine\tPRON\nwelt\tN\n<SENT>\n"
# #                                        "hey\tBEGR\ndies\tART\nsein\tVERB\ndeine\tPRON\n<SENT>\n"
# #                                        "hey\tBEGR\ndu\tPRON\nMensch\tN\n<SENT>",
# #              language=Language.DE, title="D1")
# # print(d.get_flat_document_tokens(lemma=False, lower=True))
# # print(d.get_corpus_vocab())


ds = [Document.create_document_from_doc_file(
    'E:\ma-doc-embeddings\corpora\german_series\gs_20_1_-_Karl_Bleibtreu_-_BismarckBand2_-_de_-_None_-_1915.txt') for i
      in tqdm(range(100))]
print(ds[0].get_sentence_nr())
# print(d.lemma)
