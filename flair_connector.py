from typing import Union

from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings, DocumentPoolEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from tensorflow import Tensor

from corpus_iterators import FlairDocumentIterator, FlairFacetIterator
from corpus_structure import Corpus


class FlairConnector:
    def __init__(self, word_embedding_base: str = None, document_embedding: str = None):
        """

        :param word_embedding_base: - glove: 'glove', (only en), - fasttext: 'en', 'de'
        :param document_embedding:  pool vs rnn for w2v mode - bert: 'bert', 'bert-de'  - 'longformer' (only en) -
        'flair', 'stacked-flair', 'flair-de', 'stacked-flair-de'
        """

        # document embedding

        if word_embedding_base:
            self.word_embedding_base = WordEmbeddings(word_embedding_base)

            if document_embedding.lower() == 'pool':
                self.document_embedding = DocumentPoolEmbeddings([self.word_embedding_base])
            elif document_embedding.lower() == 'rnn':
                self.document_embedding = DocumentRNNEmbeddings([self.word_embedding_base])
            else:
                raise UserWarning(f'{document_embedding} is not supported for combination with word embeedings')
        elif document_embedding:
            if document_embedding.lower() == 'bert':
                self.document_embedding = TransformerDocumentEmbeddings('bert-base-cased')
            elif document_embedding.lower() == 'bert-de':
                self.document_embedding = TransformerDocumentEmbeddings('bert-base-german-cased')
            elif document_embedding.lower() == 'longformer':
                self.document_embedding = TransformerDocumentEmbeddings('allenai/longformer-base-4096')
            elif document_embedding.lower() == 'xlnet':
                self.document_embedding = TransformerDocumentEmbeddings('xlnet-base-cased')
            elif document_embedding.lower() == 'xlnet-de':
                self.document_embedding = TransformerDocumentEmbeddings('xlm-mlm-ende-1024')
            elif document_embedding.lower() == 'flair':
                self.document_embedding = FlairEmbeddings('en-forward')
            elif document_embedding.lower() == 'flair-de':
                self.document_embedding = FlairEmbeddings('de-forward')
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

    def embedd_document(self, document: str) -> Tensor:
        flair_doc = Sentence(document)

        self.document_embedding.embed(flair_doc)
        return flair_doc.get_embedding().detach().numpy()

    def embedd_documents(self, documents: Union[FlairDocumentIterator, FlairFacetIterator]):
        return {doc_id: self.embedd_document(document) for doc_id, document in documents}


if __name__ == "__main__":
    docs = FlairDocumentIterator(Corpus.fast_load(path="corpora/german_series_all_no_limit_no_filter_real_",
                                                  load_entities=False))

    flair_instance = FlairConnector(word_embedding_base=None, document_embedding='flair-de')
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
