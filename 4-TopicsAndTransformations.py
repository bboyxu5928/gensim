#encoding:utf-8
#主题与转换（Topics and Transformations）
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#转换接口
from gensim import corpora,models,similarities
dictionary = corpora.Dictionary.load('./tmp/deerwester.dict')
corpus = corpora.MmCorpus('./tmp/corpus.mm')
for line in dictionary:
    print(line)
for doc in corpus:
    print(doc)
print(corpus)
