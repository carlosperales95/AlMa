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
        evidences.append(jso[idx]['text'])

    f.close()



from gensim.models import Phrases
sentence_stream = [doc.split(" ") for doc in evidences]

#sentence_stream=brown_raw[0:10]
bigram = Phrases(sentence_stream, min_count=1, delimiter=b' ')
trigram = Phrases(bigram[sentence_stream], min_count=1, delimiter=b' ')
bigrams_ = []
trigrams_ = []

for sent in sentence_stream:
    bigrams_.append([b for b in bigram[sent] if b.count(' ') == 1])
    trigrams_.append([t for t in trigram[bigram[sent]] if t.count(' ') == 2])

    #print(bigrams_)
    #print(trigrams_)

#NEED TO CLEAN ARGS, BIGRAMS ARE TOO WEIRD
import gzip
import gensim
import logging
import re



bis = [['how', 'is', 'tell', 'al.'], [], ['( 2010', 'et al.']]

#clean (can be done better)
for bi in bigrams_:
    for idx, b in enumerate(bi):
        date = re.search('[0-9]*[0-9][0-9]', b)
        if date is not None:
            #print("remove " + bi[idx])
            bi.remove(bi[idx])

for bi in bigrams_:
    for idx, b in enumerate(bi):
        if b.find("al.") != -1:
            bi.remove(bi[idx])
        if b.find(" =") != -1:
            bi.remove(bi[idx])
        if len(b) == 1:
            bi.remove(bi[idx])
        if b.find(" et") != -1:
            bi.remove(bi[idx])
        if b.find("-- ") != -1:
            bi.remove(bi[idx])
        if b.find("( ") != -1 or b.find(" )") != -1:
            bi.remove(bi[idx])



#print(bigrams_)
#print(bigrams_)


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

print("TRAINED")
#print(model.wv.vocab)
#print(model.wv.most_similar(positive="machine"))

from nltk.cluster import KMeansClusterer
import nltk
from sklearn import cluster
from sklearn import metrics

w2v_vectors = model.wv.vectors # here you load vectors for each word in your model
w2v_indices = {word: model.wv.vocab[word].index for word in model.wv.vocab} # here you load indices - with whom you can find an index of the particular word in your model

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


#X = model.wv.vocab
#for evidence in evidences:
#    #print(evidence)
#    for i in sent_tokenize(evidence):
#        #print(i)
#        if len(i) > 10:
#            #print(sent_tokenize(i))
#            X.append(word_tokenize(i))


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

def isImportant(label, mentions):
    important = False
    label = label.split(' ')
    for lab in label:
        for mention in mentions:
            if mention.text.find(lab) != -1:
                important = True
                break
    return important


def tsne_plot2(model, mentions):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        if isImportant(word, mentions) == True:
            tokens.append(model[word])
            labels.append(word)
            #print(word)

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


for idx, m in enumerate(mentions):
    date = re.search('[0-9]*[0-9][0-9]', str(m))
    if date is not None:
        #print("remove " + bi[idx])
        mentions.remove(mentions[idx])

for idx, m in enumerate(mentions):
    if str(m).find("al.") != -1:
        mentions.remove(mentions[idx])
    if str(m).find(" =") != -1:
        mentions.remove(mentions[idx])
    if len(str(m)) == 1:
        mentions.remove(mentions[idx])
    if str(m).find(" et") != -1:
        mentions.remove(mentions[idx])
    if str(m).find("-- ") != -1:
        mentions.remove(mentions[idx])
    if str(m) == '':
        mentions.remove(mentions[idx])
    if str(m).find("( ") != -1 or b.find(" )") != -1:
        mentions.remove(mentions[idx])

for idx, w in enumerate(mentions):
    if w in set(stopwords.words('english')):
        mentions.remove(mentions[idx])

for mention in mentions:
    print(mention)



tsne_plot(model)
tsne_plot2(model, mentions)
