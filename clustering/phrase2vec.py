from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import warnings

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


import random
import json
import sys
from os import listdir
from os.path import isfile, join
import os

from nltk.cluster import KMeansClusterer
import nltk
from sklearn import cluster
from sklearn import metrics
import numpy as np
import spacy
from word2utils import *

from gensim.models import Phrases

import gzip
import gensim
import logging
import re



dir = "./rank/batch_503/"


print("Joining evidences....................")


nlp_base = spacy.load("en_core_web_sm")
nlp = spacy.load('mymodel')


evidences, claims = getEvidences(dir)

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:
    return None

def lemmatize_sentence(sentence):
  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
  wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)

  res_words = []
  for word, tag in wn_tagged:
    if tag is None:
      res_words.append(word)
    else:
      res_words.append(lemmatizer.lemmatize(word, tag))

  return " ".join(res_words)

evid=[]
for e in evidences:
    evid.append(lemmatize_sentence(e))

#print(evid)

bigrams_, trigrams_ = evid2bitriGrams(evid)


trigrams_ = filterStringRubble(trigrams_)

#NEED TO CLEAN ARGS, BIGRAMS ARE TOO WEIRD

#print(bigrams_)
#print(bigrams_)

print("(word2vec) Finding for bigrams and trigrams....................")


#print(bigram[evidences])
model = gensim.models.Word2Vec(
        bigrams_,
        size=150,
        window=10,
        min_count=2,
        workers=10,
        iter=10)

model2 = gensim.models.Word2Vec(
        trigrams_,
        size=150,
        window=10,
        min_count=2,
        workers=10,
        iter=10)



#print(model.wv.vocab)
#print(model.wv.most_similar(positive="machine"))

#w2v_vectors = model.wv.vectors # here you load vectors for each word in your model
#w2v_indices = {word: model.wv.vocab[word].index for word in model.wv.vocab} # here you load indices - with whom you can find an index of the particular word in your model



#mentions = filterDates(mentions)
#mentions = filterWeirdChars(mentions)

print("Finding Technology/Method mentions....................")


mentions = mentionsFromArgs(evid, nlp, nlp_base)
mentions = filterWeirdChars(mentions)
mentions = filterStopwords(mentions)
mentions = filterSingleStrings(mentions)
true_mentions = filterDoubles(mentions)

semantic_mentions = getSemanticMentions(evid)

true_mentions = filterSingleChars(true_mentions)
semantic_mentions = filterSingleChars(semantic_mentions)

pointed_mentions = addPoints(true_mentions, semantic_mentions, 1.3, 1.6)

pointed_mentions = sorted(pointed_mentions, key=lambda tup: tup[1], reverse=True)


f = open('./pointed_mentions.txt', "w")
f.write("List of scored Method/Technologies")
f.write("\n")
f.write("-----------------------------------")
f.write("\n")
f.write("\n")

for p in pointed_mentions:
    f.write(str(p))
    f.write("\n")

f.close()

print("Clustering/Plotting Bigrams....................")


#filterW2VSoft/HArd
vectors2, labels2 = filterW2VSoft(model2, pointed_mentions)


import numpy as np
from nltk.cluster import KMeansClusterer, euclidean_distance

w2v_vectors = model2.wv.vectors
# here you load indices - with whom you can find an index of the particular word in your model
w2v_indices = {word: model2.wv.vocab[word].index for word in model2.wv.vocab}



# test k-means using 2 means, euclidean distance, and 10 trial clustering repetitions with random seeds
clusterer = KMeansClusterer(4, euclidean_distance, repeats=10)
clusters = clusterer.cluster(vectors2, True)
centroids = clusterer.means()
print('Clustered ')
print('As:', clusters)
#print('Means:', centroids)



# classify a new vector
#vector = np.array([2,2])
#print('classify(%s):' % vector, end=' ')
#print(clusterer.classify(vector))



import matplotlib.pyplot as plt

x0 = np.array([x[0] for idx, x in enumerate(vectors2) if clusters[idx]==0])
y0 = np.array([x[1] for idx, x in enumerate(vectors2) if clusters[idx]==0])
labels0 = np.array([labels2[idx] for idx, x in enumerate(vectors2) if clusters[idx]==0])
plt.scatter(x0,y0, color='blue')
for i, x in enumerate(x0):
    plt.annotate(labels0[i],
                 xy=(x0[i], y0[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

x1 = np.array([x[0] for idx, x in enumerate(vectors2) if clusters[idx]==1])
y1 = np.array([x[1] for idx, x in enumerate(vectors2) if clusters[idx]==1])
labels1 = np.array([labels2[idx] for idx, x in enumerate(vectors2) if clusters[idx]==1])
plt.scatter(x1,y1, color='orange')
for i, x in enumerate(x1):
    plt.annotate(labels1[i],
                 xy=(x1[i], y1[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

x2 = np.array([x[0] for idx, x in enumerate(vectors2) if clusters[idx]==2])
y2 = np.array([x[1] for idx, x in enumerate(vectors2) if clusters[idx]==2])
plt.scatter(x2,y2, color='green')
labels02 = np.array([labels2[idx] for idx, x in enumerate(vectors2) if clusters[idx]==2])
for i, x in enumerate(x2):
    plt.annotate(labels02[i],
                 xy=(x2[i], y2[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

x3 = np.array([x[0] for idx, x in enumerate(vectors2) if clusters[idx]==3])
y3 = np.array([x[1] for idx, x in enumerate(vectors2) if clusters[idx]==3])
labels3 = np.array([labels2[idx] for idx, x in enumerate(vectors2) if clusters[idx]==3])
plt.scatter(x3,y3, color='purple')
for i, x in enumerate(x3):
    plt.annotate(labels3[i],
                 xy=(x3[i], y3[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
#x4 = np.array([x[0] for idx, x in enumerate(w2v_vectors) if clusters[idx]==4])
#y4 = np.array([x[1] for idx, x in enumerate(w2v_vectors) if clusters[idx]==4])
#plt.scatter(x4,y4, color='purple')

print("Clustering/Plotting Trigrams....................")


vectors3, labels3 = filterW2VHard(model, pointed_mentions)


wv_vectors = model.wv.vectors
# here you load indices - with whom you can find an index of the particular word in your model
wv_indices = {word: model.wv.vocab[word].index for word in model.wv.vocab}




clusterer2 = KMeansClusterer(4, euclidean_distance, repeats=10)
clusters2 = clusterer2.cluster(vectors3, True)
centroids2 = clusterer2.means()
print('Clustered ')
print('As:', clusters2)

x20 = np.array([x[0] for idx, x in enumerate(vectors3) if clusters2[idx]==0])
y20 = np.array([x[1] for idx, x in enumerate(vectors3) if clusters2[idx]==0])
labels20 = np.array([labels3[idx] for idx, x in enumerate(vectors3) if clusters2[idx]==0])
plt.scatter(x20,y20, color='black')
for i, x in enumerate(x20):
    plt.annotate(labels20[i],
                 xy=(x20[i], y20[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

x21 = np.array([x[0] for idx, x in enumerate(vectors3) if clusters2[idx]==1])
y21 = np.array([x[1] for idx, x in enumerate(vectors3) if clusters2[idx]==1])
labels21 = np.array([labels3[idx] for idx, x in enumerate(vectors3) if clusters2[idx]==1])
plt.scatter(x21,y21, color='pink')
for i, x in enumerate(x21):
    plt.annotate(labels21[i],
                 xy=(x21[i], y21[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

x22 = np.array([x[0] for idx, x in enumerate(vectors3) if clusters2[idx]==2])
y22 = np.array([x[1] for idx, x in enumerate(vectors3) if clusters2[idx]==2])
plt.scatter(x2,y2, color='brown')
labels22 = np.array([labels3[idx] for idx, x in enumerate(vectors3) if clusters2[idx]==2])
for i, x in enumerate(x22):
    plt.annotate(labels22[i],
                 xy=(x22[i], y22[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

x23 = np.array([x[0] for idx, x in enumerate(vectors3) if clusters2[idx]==3])
y23 = np.array([x[1] for idx, x in enumerate(vectors3) if clusters2[idx]==3])
labels23 = np.array([labels2[idx] for idx, x in enumerate(vectors3) if clusters2[idx]==3])
plt.scatter(x23,y23, color='yellow')
for i, x in enumerate(x23):
    plt.annotate(labels23[i],
                 xy=(x23[i], y23[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')






xc = np.array([x[0] for x in centroids])
yc = np.array([x[1] for x in centroids])
plt.scatter(xc,yc, color='red')
plt.show()



doTheKM(bigrams_, model, evid, "bigram_claims.txt")
doTheKM(trigrams_, model2, evid, "trigram_claims.txt")




#print(true_mentions)

#tsne_plot(model)
#tsne_plot2(model, tris)
