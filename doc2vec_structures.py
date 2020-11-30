from typing import Dict
from gensim.models.doc2vec import Doctag
from gensim.models.keyedvectors import Doc2VecKeyedVectors, KeyedVectors, WordEmbeddingsKeyedVectors
import numpy as np


class KeyedWordVectors(WordEmbeddingsKeyedVectors):
    def __init__(self, wv: Dict[str, np.array]):
        vector_size = list(wv.values())[0].size
        super(WordEmbeddingsKeyedVectors, self).__init__(vector_size=vector_size)
        self.add(entities=list(wv.keys()), weights=np.array(list(wv.values())))


class KeyedDocumentVectors(Doc2VecKeyedVectors):
    def __init__(self, dv: Dict[str, np.array]):
        mapfile_path = None
        vector_size = list(dv.values())[0].size
        super().__init__(vector_size, mapfile_path)
        self.add(entities=list(dv.keys()), weights=np.array(list(dv.values())))
        self.vectors_docs = self.vectors
        self.vectors = []
        self.doctags = {key: Doctag(i, -1, 1) for i, key in enumerate(self.vocab.keys())}
        self.vocab = {}


class DocumentKeyedVectors:
    def __init__(self, kv: KeyedVectors,
                 prefix='*dt_'):
        wv = {}
        docvecs = {}
        for key in kv.vocab:
            if key.startswith(prefix):
                docvecs[key.replace(prefix, "")] = kv[key]
            else:
                wv[key] = kv[key]

        if len(wv) > 0:
            self.wv = KeyedWordVectors(wv)
        else:
            self.wv = {}
        self.docvecs = KeyedDocumentVectors(docvecs)


# class OriginDocumentKeyedVectors:
#     def __init__(self, kv: KeyedVectors,
#                  prefix='*dt_'):
#         # print(kv.vocab)
#         # print(kv["*dt_bs_96_loc"])
#         self.wv = {}
#         self.docvecs = {}
#         for key in kv.vocab:
#             if key.startswith(prefix):
#                 self.docvecs[key.replace(prefix, "")] = kv[key]
#             else:
#                 self.wv[key] = kv[key]
