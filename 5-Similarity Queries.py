#encoding:utf-8
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora,models,similarities
dictionary = corpora.Dictionary.load('./tmp/deerwester.dict')
corpus = corpora.MmCorpus('./tmp/deerwester.mm')
print(corpus)
lsi = models.LsiModel(corpus,id2word=dictionary,num_topics=2)
doc = "Human computer interation"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] #convert the query to LSI space
print(vec_lsi)
#初始化查询结构
index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it 
index.save('./tmp/deerwester.index')
index = similarities.MatrixSimilarity.load('./tmp/deerwester.index')
#实施查询
sims = index[vec_lsi] # perform a similarity query against the corpus
print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-

sims = sorted(enumerate(sims),key = lambda item: -item[1])
print(sims)