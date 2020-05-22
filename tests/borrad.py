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
