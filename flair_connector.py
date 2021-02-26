import multiprocessing
import os
from typing import Union, Tuple

from flair.data import Sentence
from flair.datasets import TREC_6
from flair.embeddings import TransformerDocumentEmbeddings, DocumentPoolEmbeddings, FlairEmbeddings, StackedEmbeddings, \
    SentenceTransformerDocumentEmbeddings, TransformerWordEmbeddings
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier, LanguageModel
from flair.trainers import ModelTrainer
from flair.trainers.language_model_trainer import LanguageModelTrainer
from joblib import Parallel, delayed
from tensorflow import Tensor
from torch.optim.adam import Adam
from tqdm import tqdm

from corpus_iterators import FlairDocumentIterator, FlairFacetIterator
from corpus_structure import Corpus
from sentence_transformers import SentenceTransformer

class FlairConnector:
    def __init__(self, word_embedding_base: str = None, document_embedding: str = None, fine_tune: bool = False,
                 pretuned: bool = False):
        """

        :param word_embedding_base: - glove: 'glove', (only en), - fasttext: 'en', 'de'
        :param document_embedding:  pool vs rnn for w2v mode - bert: 'bert', 'bert-de'  - 'longformer' (only en) -
        'flair', 'stacked-flair', 'flair-de', 'stacked-flair-de'
        """
        # document embedding
        self.fine_tune = fine_tune
        if word_embedding_base:
            self.word_embedding_base = WordEmbeddings(word_embedding_base)

            if document_embedding.lower() == 'pool':
                self.document_embedding = DocumentPoolEmbeddings([self.word_embedding_base])
            elif document_embedding.lower() == 'rnn':
                self.document_embedding = DocumentRNNEmbeddings([self.word_embedding_base])
            else:
                raise UserWarning(f'{document_embedding} is not supported for combination with word embeedings')
        elif document_embedding:
            print(document_embedding)
            if pretuned:
                if document_embedding.lower() == 'bert':
                    self.document_embedding = SentenceTransformer('stsb-bert-large')
                    # self.document_embedding = SentenceTransformerDocumentEmbeddings('stsb-bert-large')
                elif document_embedding.lower() == 'roberta':
                    self.document_embedding = SentenceTransformer('stsb-roberta-large')
                    # self.document_embedding = SentenceTransformerDocumentEmbeddings('stsb-roberta-large')
                elif document_embedding.lower() == 'xlm':
                    self.document_embedding = SentenceTransformer('stsb-xlm-r-multilingual')
                    # self.document_embedding = SentenceTransformerDocumentEmbeddings('stsb-xlm-r-multilingual')
            else:
                if document_embedding.lower() == 'bert':
                    self.document_embedding = TransformerDocumentEmbeddings('bert-base-cased', fine_tune=fine_tune)
                elif document_embedding.lower() == 'bert-de':
                    self.document_embedding = TransformerDocumentEmbeddings('bert-base-german-cased',
                                                                            fine_tune=fine_tune)
                elif document_embedding.lower() == 'longformer':
                    self.document_embedding = TransformerDocumentEmbeddings('allenai/longformer-base-4096',
                                                                            fine_tune=fine_tune)
                elif document_embedding.lower() == 'xlnet':
                    self.document_embedding = TransformerDocumentEmbeddings('xlnet-base-cased', fine_tune=fine_tune)
                elif document_embedding.lower() == 'xlnet-de':
                    self.document_embedding = TransformerDocumentEmbeddings('xlm-mlm-ende-1024', fine_tune=fine_tune)
                elif document_embedding.lower() == 'flair':
                    self.document_embedding = FlairEmbeddings('en-forward', fine_tune=fine_tune)
                elif document_embedding.lower() == 'flair-de':
                    self.document_embedding = FlairEmbeddings('de-forward', fine_tune=fine_tune)
                elif document_embedding.lower() == 'stack-flair':
                    self.document_embedding = StackedEmbeddings([
                        FlairEmbeddings('en-forward'),
                        FlairEmbeddings('en-backward'),
                    ])
                elif document_embedding.lower() == 'stack-flair-de':
                    self.document_embedding = StackedEmbeddings([
                        FlairEmbeddings('de-forward'),
                        FlairEmbeddings('de-backward'),
                    ])
        else:
            raise UserWarning(f'No embeddings defined')

    def fine_tune(self):
        if isinstance(self.document_embedding, TransformerDocumentEmbeddings):
            corpus = TREC_6()
            label_dict = corpus.make_label_dictionary()
            classifier = TextClassifier(self.document_embedding, label_dictionary=label_dict)
            trainer = ModelTrainer(classifier, corpus, optimizer=Adam)

            # 6. start the training
            trainer.train('resources/taggers/trec',
                          learning_rate=3e-5,  # use very small learning rate
                          mini_batch_size=16,
                          mini_batch_chunk_size=4,  # optionally set this if transformer is too much for your machine
                          max_epochs=5,  # terminate after 5 epochs
                          )
        else:
            raise UserWarning("No fine tuning for this embedding type implemented")

    def ft(self):
        if isinstance(self.document_embedding, LanguageModel):
            trainer = LanguageModelTrainer(self.document_embedding, corpus)
            trainer.train('resources/taggers/language_model',
                          sequence_length=100,
                          mini_batch_size=100,
                          learning_rate=20,
                          patience=10,
                          checkpoint=True)

    def embedd_document(self, document: str) -> Tensor:
        flair_doc = Sentence(document)

        self.document_embedding.embed(flair_doc)
        return flair_doc.get_embedding().detach().numpy()

    def embedd_document_p(self, document: str, doc_id: str) -> Tuple[Tensor, str]:
        flair_doc = Sentence(document)

        self.document_embedding.embed(flair_doc)
        return flair_doc.get_embedding().detach().numpy(), doc_id

    def embedd_documents(self, documents: Union[FlairDocumentIterator, FlairFacetIterator]):
        parallel = False
        doc_bar = tqdm(documents, total=len(documents), desc="Flair Embedding", disable=True)
        if parallel:
            num_cores = int(0.75 * multiprocessing.cpu_count())
            print(f"parralized on {num_cores} cores")
            result_tuples = Parallel(n_jobs=num_cores)(delayed(self.embedd_document_p)(document, doc_id)
                                                       for doc_id, document in doc_bar)
            return {doc_id: doc_vec for (doc_vec, doc_id) in result_tuples}
        else:
            if isinstance(self.document_embedding, SentenceTransformer):
                embeddings = self.document_embedding.encode([document for doc_id, document in doc_bar], batch_size=10,
                                                            show_progress_bar=True)
                doc_ids = (doc_id for doc_id, document in doc_bar)
                return {doc_id: embedding
                        for embedding, doc_id in zip(embeddings, doc_ids)}
            else:
                # sentences = [Sentence(document) for doc_id, document in doc_bar]
                # self.document_embedding.embed(sentences)
                # doc_ids = (doc_id for doc_id, document in doc_bar)
                # return {doc_id: sentence.get_embedding().detach().numpy()
                #         for sentence, doc_id in zip(sentences, doc_ids)}
                # print(len(self.document_embedding.embeddings))
                return {doc_id: self.embedd_document(document) for doc_id, document in doc_bar}


if __name__ == "__main__":
    # docs = FlairDocumentIterator(Corpus.fast_load(path="corpora/german_series_all_no_limit_no_filter_real_",
    #                                               load_entities=False))

    # docs = FlairDocumentIterator(Corpus.fast_load(path="corpora/classic_gutenberg_all_no_limit_no_filter_real_",
    #                                               load_entities=False))
    #
    # flair_instance = FlairConnector(word_embedding_base=None, document_embedding='bert', pretuned=True)
    # # d = flair_instance.embedd_documents(docs)
    # # print(d.keys())
    # # print(flair_instance.embedd_document(doc))
    # for document_id, doc in docs:
    #     print(document_id)
    #     print(flair_instance.embedd_document(doc).shape)
    # doc = 'The grass is green . And the nice sky is blue .'*10

    # from flair.data import Corpus
    # from flair.datasets import ColumnCorpus
    #
    # columns = {0: 'text', 1: 'lemma', 2: 'pos', 3: 'ner', 4: 'punct', 5: 'alpha', 6: 'stop', 7: 'doc_id'}
    #
    # # this is the folder in which train, test and dev files reside
    # data_folder = 'corpora/classic_gutenberg_all_no_limit_no_filter_real__flair_text'
    #
    # # init a corpus using column format, data folder and the names of the train, dev and test files
    # # corpus: Corpus = ColumnCorpus(data_folder, columns)
    # corpus: Corpus = ColumnCorpus(data_folder, columns)
    # print(len(corpus.train))
    #
    # print(corpus.train[0].to_tagged_string('ner'))
    # print(corpus.train[1].to_tagged_string('pos'))
    #
    # label_dict = corpus.make_label_dictionary()
    #
    # language_model = FlairEmbeddings('news-forward').lm
    #
    # # classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)
    # # trainer = ModelTrainer(classifier, corpus)
    #
    # trainer = ModelTrainer(language_model, corpus)
    #
    # trainer.train('resources/taggers/language_model',
    #               mini_batch_size=100,
    #               learning_rate=20,
    #               patience=10,
    #               checkpoint=True)

    corpus_dir = 'corpora/classic_gutenberg_all_no_limit_no_filter_real__flair_text'

    from flair.data import Dictionary
    from flair.embeddings import FlairEmbeddings
    from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

    # instantiate an existing LM, such as one from the FlairEmbeddings
    # language_model = FlairEmbeddings('news-forward').lm
    language_model = TransformerWordEmbeddings('roberta-base', fine_tune=True)

    # are you fine-tuning a forward or backward LM?
    is_forward_lm = language_model.is_forward_lm

    # get the dictionary from the existing language model
    dictionary: Dictionary = language_model.dictionary

    # get your corpus, process forward and at the character level
    corpus = TextCorpus(corpus_dir,
                        dictionary,
                        is_forward_lm,
                        character_level=True)

    # use the model trainer to fine-tune this model on your corpus
    trainer = LanguageModelTrainer(language_model, corpus)

    trainer.train('resources/taggers/language_model',
                  sequence_length=100,
                  mini_batch_size=100,
                  learning_rate=20,
                  patience=10,
                  checkpoint=True)

    # flair_instance = FlairEmbedding(word_embedding_base='glove', document_embedding='pool')
    # print(flair_instance.embedd_document(doc))
    #
    # flair_instance = FlairEmbedding(word_embedding_base='de', document_embedding='pool')
    # print(flair_instance.embedd_document(doc))
    #
    # flair_instance = FlairEmbedding(word_embedding_base='en', document_embedding='pool')
    # print(flair_instance.embedd_document(doc))
    #
    # flair_instance = FlairEmbedding(word_embedding_base='glove', document_embedding='rnn')
    # print(flair_instance.embedd_document(doc))
    #
    # flair_instance = FlairEmbedding(word_embedding_base='de', document_embedding='rnn')
    # print(flair_instance.embedd_document(doc))
    #
    # flair_instance = FlairEmbedding(word_embedding_base='en', document_embedding='rnn')
    # print(flair_instance.embedd_document(doc))
    #
    # flair_instance = FlairEmbedding(word_embedding_base=None, document_embedding='bert')
    # print(flair_instance.embedd_document(doc))
    #
    # flair_instance = FlairEmbedding(word_embedding_base=None, document_embedding='bert-de')
    # print(flair_instance.embedd_document(doc))

    # flair_instance = FlairEmbedding(word_embedding_base=None, document_embedding='xlnet')
    # print(flair_instance.embedd_document(doc))
    #
    # flair_instance = FlairEmbedding(word_embedding_base=None, document_embedding='xlnet-de')
    # print(flair_instance.embedd_document(doc))
    #
    # sentence = Sentence('The grass is green . And the nice sky is blue .'*10000)
    # glove_embedding = WordEmbeddings('glove')
    # w_embedding = FastTextEmbeddings
    #
    # document_embeddings = DocumentRNNEmbeddings([glove_embedding])
    #
    #
    # # init embedding
    # embedding = TransformerDocumentEmbeddings('bert-base-uncased')
    #
    #
    # # embed the sentence
    # embedding.embed(sentence)
    # embedding.embed(sentence)
    #
    # print(sentence.get_embedding())
