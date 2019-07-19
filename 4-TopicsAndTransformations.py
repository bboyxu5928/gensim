#encoding:utf-8
#主题与转换（Topics and Transformations）
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#转换接口
from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('./tmp/deerwester.dict')
corpus = corpora.MmCorpus('./tmp/deerwester.mm')
print(corpus)

#创建一个转换
tfidf = models.TfidfModel(corpus)#第一步，初始化一个模型

doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow])  # 第二步 -- 使用模型转换向量

#对整个语料库实施转换
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
     print(doc)
     
     #转换也可以被序列化，还可以一个（转换）叠另一个，像一串链条一样：
     #潜在语义索引（LSI）
lsi = models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=2)#初始化一个lsi转换
corpus_lsi = lsi[corpus_tfidf] # 在原始语料库上加上双重包装: bow->tfidf->fold-in-lsi
lsi.print_topics(2)

for doc in corpus_lsi:
     print(doc)

lsi.save('./tmp/model.lsi')
lsi = models.LsiModel.load('./tmp/model.lsi')

#词频-逆文档频（Term Frequency * Inverse Document Frequency， Tf-Idf）
model = models.tfidfmodel.TfidfModel(bow_corpus, normalize=True)

#潜在语义索引（Latent Semantic Indexing，LSI，or sometimes LSA）
model = models.lsimodel.LsiModel(tfidf_corpus,id2word=dictionary,num_topics=300)

model.add_documents(another_tfidf_corpus) # 现在LSI已经使用tfidf_corpus + another_tfidf_corpus进行过训练了
lsi_vec = model[tfidf_vec] # 将新文档转化到LSI空间不会影响该模型

model.add_documents(more_documents) # tfidf_corpus + another_tfidf_corpus + more_documents
lsi_vec = model[tfidf_vec]

#随机映射（Random Projections，RP）
model = models.rpmodel.RpModel(tfidf_corpus,num_topics=500)

#隐含狄利克雷分配（Latent Dirichlet Allocation, LDA）
model = models.ldamodel.LdaModel(bow_corpus,id2word=dictionary,num_topics=100)
#分层狄利克雷过程（Hierarchical Dirichlet Process，HDP）
model = hdpmodel.HdpModel(bow_corpus, id2word=dictionary)