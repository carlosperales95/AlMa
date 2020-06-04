from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import warnings

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering



import random
import json
import sys
from os import listdir
from os.path import isfile, join
import os

import numpy as np
import spacy
import nltk

from word2utils import *




#################### MAIN CODE ######################

dir = "./rank/batch_503/"

nlp_base = spacy.load("en_core_web_sm")
nlp = spacy.load('mymodel')

evidences = getEvidences(dir)
data = sentenceTokenizer(evidences)


# Create CBOW model
model = gensim.models.Word2Vec(data, min_count = 1,
                              size = 100, window = 10)

# Print results
print("Cosine similarity between 'machine' " +
               "and 'algorithm' - CBOW : ",
    model.similarity('machine', 'algorithm'))

print("Cosine similarity between 'machine' " +
                 "and 'translation' - CBOW : ",
      model.similarity('machine', 'translation'))

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100,
                                             window = 5, sg = 1)

# Print results
print("Cosine similarity between 'machine' " +
               "and 'algorithm' - CBOW : ",
    model2.similarity('machine', 'algorithm'))

print("Cosine similarity between 'machine' " +
                 "and 'translation' - CBOW : ",
      model2.similarity('machine', 'translation'))






Z=hierarchy.linkage(X, 'ward')
dn = hierarchy.dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Euclidean distances')
plt.xlim(0,2000)
plt.ylim(0, 5)
#plt.show()

hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
print(y_hc)
#for index, sentence in enumerate(sentences):
#    print(str(y_hc[index]) + ":" + str(sentence))
