#encoding:utf-8
from gensim import corpora
#(存储语料库)
corpus = [[(1,0.5)],[]]#  让一个文档为空，作为它的heck
corpora.MmCorpus.serialize('./tmp/corpus.mm',corpus)
corpora.SvmLightCorpus.serialize('./tmp/corpus.svmligth',corpus)
corpora.BleiCorpus.serialize('./tmp/corpus.lda-c',corpus)
corpora.LowCorpus.serialize('./tmp/corpus.low',corpus)

#载入语料库
corpus = corpora.MmCorpus('./tmp/corpus.mm')
print(corpus) #语料库对象是流式的，因此你不能直接将其打印出来
print(list(corpus))

for doc in corpus:
    print('====',doc)
    
#（转存语料库）
corpora.BleiCorpus.serialize('./tmp/corpus.lda-c',corpus)

#Gensim包含了许多高效的工具函数来帮你实现语料库与numpy矩阵之间互相转换：

corpus = gensim.matutils.Dense2Corpus(numpy_matrix)

numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=number_of_corpus_features)
print('')


