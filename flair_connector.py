from typing import Union

from flair.data import Sentence
from flair.datasets import TREC_6
from flair.embeddings import TransformerDocumentEmbeddings, DocumentPoolEmbeddings, FlairEmbeddings, StackedEmbeddings, \
    SentenceTransformerDocumentEmbeddings
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from tensorflow import Tensor
from torch.optim.adam import Adam

from corpus_iterators import FlairDocumentIterator, FlairFacetIterator
from corpus_structure import Corpus


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
            if pretuned:
                if document_embedding.lower() == 'bert':
                    self.document_embedding = SentenceTransformerDocumentEmbeddings('stsb-bert-large')
                elif document_embedding.lower() == 'roberta':
                    self.document_embedding = SentenceTransformerDocumentEmbeddings('stsb-roberta-large')
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

    def embedd_document(self, document: str) -> Tensor:
        flair_doc = Sentence(document)

        self.document_embedding.embed(flair_doc)
        return flair_doc.get_embedding().detach().numpy()

    def embedd_documents(self, documents: Union[FlairDocumentIterator, FlairFacetIterator]):
        return {doc_id: self.embedd_document(document) for doc_id, document in documents}


if __name__ == "__main__":
    # docs = FlairDocumentIterator(Corpus.fast_load(path="corpora/german_series_all_no_limit_no_filter_real_",
    #                                               load_entities=False))
    docs = FlairDocumentIterator(Corpus.fast_load(path="corpora/classic_gutenberg_all_no_limit_no_filter_real_",
                                                  load_entities=False))

    flair_instance = FlairConnector(word_embedding_base=None, document_embedding='bert', pretuned=True)
    # d = flair_instance.embedd_documents(docs)
    # print(d.keys())
    # print(flair_instance.embedd_document(doc))
    for document_id, doc in docs:
        print(document_id)
        print(flair_instance.embedd_document(doc).shape)
    doc = 'The grass is green . And the nice sky is blue .'*10

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
