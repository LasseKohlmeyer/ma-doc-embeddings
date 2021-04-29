from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('stsb-roberta-large')
model = SentenceTransformer('stsb-xlm-r-multilingual')

# Our sentences we like to encode
# sentences = ['This framework generates embeddings for each input sentence',
#              'Sentences are passed as a list of string.',
#              'The quick brown fox jumps over the lazy dog.']
sentences = ['Dieses Framework generiert Einbettungen für jeden Eingabesatz',
             'Sätze werden als Liste von Strings übergeben.',
             'Der schnelle braune Fuchs springt über den faulen Hund.']

# Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

# Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
