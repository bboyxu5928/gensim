#encoding:utf-8
#logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)

from gensim import corpora,models,similarities

corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
          [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
          [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
           [(0, 1.0), (4, 2.0), (7, 1.0)],
           [(3, 1.0), (5, 1.0), (6, 1.0)],
           [(9, 1.0)],
           [(9, 1.0), (10, 1.0)],
           [(9, 1.0), (10, 1.0), (11, 1.0)],
           [(8, 1.0), (10, 1.0), (11, 1.0)],
           #放置一个一模一样的
           [(0,1),(4,1)]]
tfidf = models.TfidfModel(corpus)
vec = [(0,1),(4,1)]
#print(corpus [vec])
print('\n')
print(tfidf[vec])

print('\n')
index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=12)
sims = index[tfidf[vec]]
print(list(enumerate(sims)))