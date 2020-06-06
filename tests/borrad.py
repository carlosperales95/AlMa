text = sample_file.read()


nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

for ent in doc.ents:
  print(ent.label_, ' | ', ent.text)


print("\n")
print("\n")
print("\n")




nlp = spacy.load('mymodel')
doc = nlp(text)

for ent in doc.ents:
  print(ent.label_, ' | ', ent.text)


print("\n")
print("\n")
print("\n")


            for idy, ent in enumerate(doc.ents):
                    if ent[idy].text.find(texts[idx+1]):
                        if hasattr(ent[idy], "label_"):
                            f.write(evidence)
                            f.write("\n")
                            f.write(str(tokens))
                            f.write("\n")
                            f.write(ent[idy].label_ + "|" + ent[idy].text)
                            f.write("\n")
                            f.write("\n")
                            f.write("\n")


    sentence = paper_abstract.lower()

    # onegrams = OneGramDist(filename='count_10M_gb.txt')
    onegrams = OneGramDist(filename='count_1M_gb.txt.gz')
    # onegrams = OneGramDist(filename='count_1w.txt')
    onegram_fitness = functools.partial(onegram_log, onegrams)
    paper_abstract = segment(sentence, word_seq_fitness=onegram_fitness)









from nltk.tokenize import sent_tokenize, word_tokenize
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


import re
import logging
import time
import config

from operator import add

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.broadcast import _broadcastRegistry



dir = "./rank/batch_3068/"


onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
evidences = []
paper_evidences = []
paper_claims = []


for file in onlyfiles:

    if file.startswith( 'evidence' ):
        paper_evidences.append(file)

for idx,file in enumerate(paper_evidences):

    f = open(dir+file, "r")
    paper_full =f.read()

    jso = json.loads(paper_full)
    for idx,item in enumerate(jso):
        evidences.append(jso[idx]['text'])

    f.close()


from gensim.models import Phrases
sentence_stream = [doc.split(" ") for doc in evidences]

#sentence_stream=brown_raw[0:10]
bigram = Phrases(sentence_stream, min_count=1, delimiter=b' ')
trigram = Phrases(bigram[sentence_stream], min_count=1, delimiter=b' ')

for sent in sentence_stream:
    bigrams_ = [b for b in bigram[sent] if b.count(' ') == 1]
    trigrams_ = [t for t in trigram[bigram[sent]] if t.count(' ') == 2]

    print(bigrams_)
    print(trigrams_)










import nltk.data
from gensim.models import word2vec
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import pandas as pd
import numpy as np
import os
import re
import logging
import sqlite3
import time
import sys
import multiprocessing
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from itertools import cycle

def clustering_on_wordvecs(word_vectors, num_clusters):
    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters = num_clusters, init='k-means++')
    idx = kmeans_clustering.fit_predict(word_vectors)

    return kmeans_clustering.cluster_centers_, idx

Z = model.wv.syn0

centers, clusters = clustering_on_wordvecs(Z, 50)
centroid_map = dict(zip(model.wv.index2word, clusters))


def get_top_words(index2word, k, centers, wordvecs):
    tree = KDTree(wordvecs)#Closest points for each Cluster center is used to query the closest 20 points to it.
    closest_points = [tree.query(np.reshape(x, (1, -1)), k=k) for x in centers]
    closest_words_idxs = [x[1] for x in closest_points]#Word Index is queried for each position in the above array, and added to a Dictionary.
    closest_words = {}
    for i in range(0, len(closest_words_idxs)):
        closest_words['Cluster #' + str(i)] = [index2word[j] for j in closest_words_idxs[i][0]]#A DataFrame is generated from the dictionary.
    df = pd.DataFrame(closest_words)
    df.index = df.index+1
    return df


top_words = get_top_words(model.wv.index2word, len(centers), centers, Z)


def display_cloud(cluster_num, cmap):
    wc = WordCloud(background_color="black", max_words=2000, max_font_size=80, colormap=cmap)
    wordcloud = wc.generate(' '.join([word for word in top_words['Cluster #' + str(cluster_num).zfill(2)]]))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('cluster_' + str(cluster_num), bbox_inches='tight')


cmaps = cycle([
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])
for i in range(50):
    col = next(cmaps)
    display_cloud(i+1, col)


    #print(labels)
    #for index, evidence in enumerate(data):
    #    print(str(labels[index]) + ":" + str(evidence))

#    print("Top terms per cluster:")
#    order_centroids = clf.cluster_centers_.argsort()[:, ::-1]
    #terms = vectorizer.get_feature_names()
#    for i in range(n_clusters):
#        print ("Cluster %d: " + str(i))
#        for ind in order_centroids[i, :4]:
#            print(lr[ind])
        #print
