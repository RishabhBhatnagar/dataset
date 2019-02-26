from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts

def doc2vec(paragraphs):
    for label, para in enumerate(paragraphs):
        yield TaggedDocument(para, [label])

data = list(doc2vec(['a', 'b']))
print(data)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model = Doc2Vec(documents, vector_size=10, window=2, min_count=1, workers=4)

model = Doc2Vec(alpha=0.025, min_alpha=0.025)
model.build_vocab(data)
for epoch in range(10):
    model.train(data)
    model.alpha -= 0.002
    model.min_alpha = alpha
