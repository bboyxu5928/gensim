#encoding:utf-8
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#语料库与向量空间
from gensim import corpora,models,similarities
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
            "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]
# 去除停用词并分词
    # 译者注：这里只是例子，实际上还有其他停用词
    #         处理中文时，请借助 Py结巴分词 https://github.com/fxsjy/jieba
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]
# 去除仅出现一次的单词
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token]+=1
texts = [[token for token in text if frequency[token]>1]
         for text in texts]
from pprint import pprint
pprint(texts)

dictionary = corpora.Dictionary(texts)
#dictionary.save('../gensim/tmp/deerwester.dict') # 把字典保存起来，方便以后使用
#dictionary.save('./tmp/deerwester.dict') # 把字典保存起来，方便以后使用
dictionary.save('/deerwester.dict') # 把字典保存起来，方便以后使用


print(dictionary)
#Dictionary(12 unique tokens)
print(dictionary.token2id)

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/deerwester.mm',corpus)#存入硬盘。
print(corpus)

class MyCorpus(object):
    def __iter__(self):
        for line in open('mycorpus.txt'):
            yield dictionary.doc2bow(line.lower().split())
corpus_memory_friendly = MyCorpus() # 没有将整个语料库载入内存
print(corpus_memory_friendly)

for vector in corpus_memory_friendly: # 一次读入内存一个向量
    print(vector)

#收集所有字符的统计信息
dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))    
#收集停用词和仅出现一次的词的id
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist 
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid,docfreq in dictionary.dfs.items() if docfreq == 1]
dictionary.filter_tokens(stop_ids+once_ids) #删除停用词和仅出现一次的词
dictionary.compactify() # 消除id序列在删除词后产生的不连续的缺口
print(dictionary)