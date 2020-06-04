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

import numpy as np
import spacy
import nltk

from word2utils import *

from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering



def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


def tsne_plotMentions(model, mentions):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        for mention in mentions:
            if len(word) > 3:
                ment = [x.lower() for x in mention]
                if word in ment:
                    double = False
                    for label in labels:
                        if label == word:
                            double = True
                    if double == False:
                        print(word)
                        tokens.append(model[word])
                        labels.append(word)

    #print(labels)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        #plt.axis([0,5,5,20])
    plt.show()


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


# here you load vectors for each word in your model
w2v_vectors = model.wv.vectors
# here you load indices - with whom you can find an index of the particular word in your model
w2v_indices = {word: model.wv.vocab[word].index for word in model.wv.vocab}


#X = []
#for i in sent_tokenize(evidence):
#    X.append(vectorize(i))

from sklearn.cluster import DBSCAN
# you can change these parameters, given just for example
##dbscan = DBSCAN(metric='cosine', eps=0.07, min_samples=3)
 # where X - is your matrix, where each row corresponds to one document (line) from the docs, you need to cluster
##cluster_labels = dbscan.fit_predict(X)


mentions = mentionsFromArgs(evidences, nlp, nlp_base)
mentions = filterStopwords(mentions)
mentions = filterSingleStrings(mentions)
true_mentions = filterDoubles(mentions)

semantic_mentions = getSemanticMentions(evidences)

true_mentions = filterSingleChars(true_mentions)
semantic_mentions = filterSingleChars(semantic_mentions)

#true_mentions = spacyBaseDoubleCheck(true_mentions)

pointed_mentions = addPoints(true_mentions, semantic_mentions, 1.3, 1.6)

pointed_mentions = sorted(pointed_mentions, key=lambda tup: tup[1], reverse=True)


tris = []
for pm in pointed_mentions:
    if pm[1] > 2:
        tris.append(pm[0])

#for men in true_mentions:
#    print(str(men) + "/" + str(len(str(men))))

import numpy as np
from nltk.cluster import KMeansClusterer, euclidean_distance

w2v_vectors = model.wv.vectors
# here you load indices - with whom you can find an index of the particular word in your model
w2v_indices = {word: model.wv.vocab[word].index for word in model.wv.vocab}

vectors2 = []
labels2 = []
for idx, word in enumerate(model.wv.vocab):
    for mention in pointed_mentions:
        if len(word) > 3:
            ment = [x.lower() for x in mention[0]]
            if word in ment:
                double = False
                for labe in labels2:
                    if labe == word:
                        double = True
                if double == False:
                    labels2.append(word)
                    vectors2.append(model.wv.vectors[idx])

print(len(w2v_vectors))
print(len(model.wv.vocab))
print(len(vectors2))

#vectors = [np.array(f) for f in [[3, 3], [1, 2], [4, 2], [4, 0], [2, 3], [3, 1]]]


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
labels02 = np.array([labels2[idx] for idx, x in enumerate(vectors2) if clusters[idx]==2])
plt.scatter(x2,y2, color='green')
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

xc = np.array([x[0] for x in centroids])
yc = np.array([x[1] for x in centroids])
plt.scatter(xc,yc, color='red')
plt.show()


doTheKM(data, model)


#tsne_plot(model)

#tsne_plotMentions(model, tris)
