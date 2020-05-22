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

dir = "./rank/batch_3003/"


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
        evidences.append(jso[idx]['evidence'])

    f.close()


data = []

for evidence in evidences:
    # iterate through each sentence in the file
    for i in sent_tokenize(evidence):
        temp = []

        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())

        data.append(temp)

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count = 1,
                              size = 100, window = 10)

# Print results
print("Cosine similarity between 'machine' " +
               "and 'algorithm' - CBOW : ",
    model1.similarity('machine', 'algorithm'))

print("Cosine similarity between 'machine' " +
                 "and 'translation' - CBOW : ",
      model1.similarity('machine', 'translation'))

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



w2v_vectors = model1.wv.vectors # here you load vectors for each word in your model
w2v_indices = {word: model1.wv.vocab[word].index for word in model1.wv.vocab} # here you load indices - with whom you can find an index of the particular word in your model


import numpy as np

def vectorize(line):
    words = []
    for word in line: # line - iterable, for example list of tokens
        try:
            w2v_idx = w2v_indices[word]
        except KeyError: # if you does not have a vector for this word in your w2v model, continue
            continue
        words.append(w2v_vectors[w2v_idx])
        if words:
            words = np.asarray(words)
            min_vec = words.min(axis=0)
            max_vec = words.max(axis=0)
            return np.concatenate((min_vec, max_vec))
        if not words:
            return None

X = []
for i in sent_tokenize(evidence):
    X.append(vectorize(i))

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(metric='cosine', eps=0.07, min_samples=3) # you can change these parameters, given just for example
cluster_labels = dbscan.fit_predict(X) # where X - is your matrix, where each row corresponds to one document (line) from the docs, you need to cluster


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

def tsne_plot2(model, mentions):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        for mention in mentions:
            if len(word) > 3:
                if mention.text.find(word) != -1:
                    double = False
                    for label in labels:
                        if label == word:
                            double = True
                    if double == False:
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


import spacy
import nltk

nlp_base = spacy.load("en_core_web_sm")
nlp = spacy.load('mymodel')

mentions = []

for idx, evidence in enumerate(evidences):

    text = word_tokenize(evidences[idx])
    tokens = nltk.pos_tag(text)

    tags = [lis[1] for lis in tokens]
    texts = [lis[0] for lis in tokens]

    precandidates = []
    candidates = []
    doc = nlp(evidence)
    entities = doc.ents
    doc_base = nlp_base(evidence)
    entities_base = doc_base.ents
    entities_final = []
    for ent in entities_base:
      for dent in entities:
          if hasattr(ent, 'label_') and hasattr(dent, 'label_'):
              if ent.text.find(dent.text):
                  if ent.label_ == 'PRODUCT' or ent.label_ == 'ORG' or ent.label_ == 'NORP':
                      entities_final.append(dent)
                  #else:
              else:
                  entities_final.append(dent)

    for idx, tag in enumerate(tags):
        if tag == 'NN' or tag.startswith('JJ'):
            #print(idx)
            precandidates.append(texts[idx])
            candidates.append(texts[idx+1])

    for candidate in candidates:
        for ent in entities:
            if hasattr(ent, 'label_'):
                if ent.text.find(candidate):
                    mentions.append(ent)


    mentions = list(dict.fromkeys(mentions))


for idx, w in enumerate(mentions):
    if w in set(stopwords.words('english')):
        mentions.remove(mentions[idx])

tsne_plot(model1)

tsne_plot2(model1, mentions)
