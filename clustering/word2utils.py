from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import warnings

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

from nltk.cluster import KMeansClusterer
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

from sklearn.feature_extraction.text import TfidfVectorizer



def getEvidences(dir):

    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

    paper_evidences = []
    paper_claims = []

    evidences = []
    claims = []


    for file in onlyfiles:

        if file.startswith( 'evidence' ):
            paper_evidences.append(file)

        if file.startswith( 'claim' ):
            paper_claims.append(file)

    for idx,file in enumerate(paper_evidences):

        f = open(dir+file, "r")
        paper_full =f.read()

        jso = json.loads(paper_full)
        for idx,item in enumerate(jso):
            if jso[idx] != "":
                evidences.append(jso[idx]['text'])

        f.close()

    for idx,file in enumerate(paper_claims):

        f = open(dir+file, "r")
        paper_full =f.read()

        jso = json.loads(paper_full)
        for idx,item in enumerate(jso):
            if jso[idx] != "":
                claims.append(jso[idx]['text'])

        f.close()


    return evidences, claims



def sentenceTokenizer(evidences):

    data = []

    for evidence in evidences:
        # iterate through each sentence in the file
        for i in sent_tokenize(evidence):
            temp = []

            # tokenize the sentence into words
            for j in word_tokenize(i):
                temp.append(j.lower())

            data.append(temp)

    return data


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


def mentionsFromArgs(evidences, nlp, nlp_base):

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
                if len(texts) > idx+1:
                    candidates.append(texts[idx+1])

        for candidate in candidates:
            for ent in entities:
                if hasattr(ent, 'label_'):
                    if ent.text.find(candidate):
                        mentions.append(ent)

    return mentions


def filterStopwords(mentions):

    for idx, w in enumerate(mentions):
        if w in set(stopwords.words('english')):
            mentions.remove(mentions[idx])

    return mentions


def filterSingleStrings(mentions):
    lenmentions = []
    for mention in mentions:
        if len(str(mention)) > 2:
            lenmentions.append(mention)

    return lenmentions


def filterDoubles(mentions):

    true_mentions = []
    for mention in mentions:
        double = False
        for tm in true_mentions:
            #print(true_mentions)
            if str(mention) == str(tm):
                #print(str(mention) + " / " + str(tm))
                double = True
                break
        if double == False:
            m_arr = str(mention).split(' ')
            if len(m_arr) < 6:
                true_mentions.append(m_arr)

    return true_mentions


def getSemanticMentions(evidences):
    semantic_mentions = []

    for idt, evidence in enumerate(evidences):

        text = word_tokenize(evidences[idt])
        tokens = nltk.pos_tag(text)

        tags = [lis[1] for lis in tokens]
        texts = [lis[0] for lis in tokens]

        #phrases = []
        continuation = 0


        for idx, tag in enumerate(tags):
            subject = False
            verbal = False
            index = []
            index3 = []
            skiptomethod = False
            methodstart = False
            index2 = []
            other = False
            proper_method = False
            methodstart_id = 0

            if continuation > 0:
                idx = continuation
            if tags[idx] == 'PRP':
                subject = True
                index.append(idx)

            elif tags[idx] == 'DT' and tags[idx+1].startswith('NN'):
                subject = True
                index.append(idx)
                index.append(idx+1)

            elif tags[idx].startswith('NN'):
                subject = True
                index.append(idx)

            end = len(index)
            if subject:
                for idy, t in enumerate(tags):
                    if idy > index[end-1]:
                        if t.startswith('VB'):
                            verbal = True
                            index2.append(idy)
                            id = idy
                            tagloop = tags[id+1:]
                            for idz, tl in enumerate(tagloop):
                                while not other:
                                    if tagloop[idz].startswith('VB'):
                                        index2.append(id + (idz + 1))
                                        idz += 1
                                    else:
                                        other = True
                                break
                            if verbal and other:
                                break
                        else:
                            verbal = False
                            break
                end2 = len(index2)
                other = False
                if subject and verbal:
                    for idz, tt in enumerate(tags):
                        if idz > index2[end2-1]:
                            if tt.startswith('JJ'):
                                methodstart = True
                                index3.append(idz)
                                break
                            elif len(tags) > (idz + 1):
                                if tt == 'DT' and ( tags[idz+1].startswith('NN') or tags[idz+1].startswith('JJ')):
                                    methodstart = True
                                    index3.append(idz)
                                    index3.append(idz+1)
                                    break
                                elif tt not in ['PRP$', 'IN']:
                                    methodstart = False
                                    break

                    if methodstart:
                        id = index3[len(index3) - 1]
                        methodstart_id = index3[len(index3) - 1]
                        tagloop = tags[id+1:]
                        for idw, tl in enumerate(tagloop):
                            while not other:
                                if tagloop[idw].startswith('NN'):
                                    proper_method = True

                                if tagloop[idw].startswith('NN') or tagloop[idw].startswith('JJ'):
                                    index3.append(id + (idw+1))
                                    if (id+idw+1) == (len(texts)-1):
                                        other = True
                                        break
                                    else:
                                        idw += 1
                                else:
                                    other = True
                                    break

            if proper_method:
                mmention = []
                for ind in index3:
                    if(ind >= methodstart_id):
                        mmention.append(texts[ind])
                if len(mmention) < 6:
                    semantic_mentions.append(mmention)

                continuation = index3[(len(index3) - 1)] + 1

    return semantic_mentions


def filterSingleChars(array):
    filter_shorts = []
    for men in array:
        mention = []
        for element in men:
            if len(str(element)) > 2:
                mention.append(element)
        if len(mention) > 0:
            filter_shorts.append(mention)

    return filter_shorts


#Work In Progress
def spacyBaseDoubleCheck(array, nlp, nlp_base):

    filtered_ents = []
    for ent in array:
        entity = ""
        for el in ent:
            entity = entity + str(el) + " "
        doc = nlp(entity)
        entities = doc.ents
        doc_base = nlp_base(entity)
        entities_base = doc_base.ents
        for nen in entities:
            for bent in entities_base:
                if hasattr(bent, 'label_') and hasattr(ent, 'label_') and (bent.text == ent.text):
                    if bent.label_ in ['PRODUCT', 'ORG', 'NORP', 'WORK_OF_ART']:
                        filtered_ents.append(ent)

                else:
                    filtered_ents.append(ent)

    return filtered_ents


def addPoints(array1, array2, points1, points2):

    target = []
    for idx, element in enumerate(array1):
        points = points1
        for idy, double_dummy in enumerate(array1):
            if idy != idx:
                #print(str(element) + "////" + str(double_dummy))
                if element == double_dummy:
                    points += 0.4

        for e in element:
            findings = 0
            for comparative in array2:
                if e in comparative:
                    findings += 1
            if findings == len(element):
                points += points2
            if len(element) == 1 and e.isupper():
                points = points * 2


        tuple = [element, points]
        target.append(tuple)

    for element in array2:
        points = points2
        for e in element:
            findings = 0
            for comparative in array2:
                if e in comparative:
                    findings += 1
            if findings == len(element):
                points += points1
        tuple = [element, points]
        target.append(tuple)

    filtered_doubles = []
    for element in target:
        double = False
        for elem in filtered_doubles:
            if element == elem:
                double = True
                break
        if double == False:
            filtered_doubles.append(element)

    return filtered_doubles



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
            if lab in mention:
                important = True
                break
    return important


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


def filterDates(mentions):
    for idx, m in enumerate(mentions):
        date = re.search('[0-9]*[0-9][0-9]', str(m))
        if date is not None:
            #print("remove " + bi[idx])
            mentions.remove(mentions[idx])
    return mentions



def evid2bitriGrams(evidences):

    sentence_stream = [doc.split(" ") for doc in evidences]

    #sentence_stream=brown_raw[0:10]
    bigram = Phrases(sentence_stream, min_count=1, delimiter=b' ')
    trigram = Phrases(bigram[sentence_stream], min_count=1, delimiter=b' ')
    bigrams_ = []
    trigrams_ = []

    for sent in sentence_stream:
        bigrams_.append([b for b in bigram[sent] if b.count(' ') == 1])
        trigrams_.append([t for t in trigram[bigram[sent]] if t.count(' ') == 2])

    return bigrams_, trigrams_



def filterStringRubble(bigrams_):

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

    return bigrams_



def filterWeirdChars(mentions):
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
        if str(m).find("( ") != -1 or str(m).find(" )") != -1:
            mentions.remove(mentions[idx])

    return mentions


def MentionRankThreshold(tris):
    tris = []
    for pm in pointed_mentions:
        if pm[1] > 2:
            tris.append(pm[0])
    return tris


def filterW2VSoft(model, pointed_mentions):
    #filter soft
    vectors2 = []
    labels2 = []
    for idx, word in enumerate(model.wv.vocab):
        label = word.split(' ')
        count = 0
        for lab in label:
            for mention in pointed_mentions:
                if len(lab) > 3:
                    ment = [x.lower() for x in mention[0]]
                    if lab in ment:
                        double = False

                        for labe in labels2:
                            if labe == word:
                                double = True
                        if double == False:
                            labels2.append(word)
                            vectors2.append(model.wv.vectors[idx])

    return vectors2, labels2


def filterW2VHard(model, pointed_mentions):
    #filter hard
    vectors = []
    labels = []
    for idx, word in enumerate(model.wv.vocab):
        label = word.split(' ')
        count = 0
        for lab in label:
            for mention in pointed_mentions:
                if len(lab) > 3:
                    ment = [x.lower() for x in mention[0]]
                    if lab in ment:
                        double = False
                        count += 1
                    else:
                        count = 0
                    if count == len(label):
                        for labe in labels:
                            if labe == word:
                                double = True
                        if double == False:
                            labels.append(word)
                            vectors.append(model.wv.vectors[idx])
        vectors2 = []
        labels2 = []
        for idx, elem in enumerate(vectors):
            count = 0
            for w in elem:
                if w not in set(stopwords.words('english')):
                    count+=1
            if count == len(elem):
                vectors2.append(elem)
                labels2.append(labels[idx])


    return vectors2, labels2


def w2vectorizer(sent, m):
    vec=[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                vec = m[w]
            else:
                vec = np.add(vec, m[w])
            numw+=1
        except:
            pass

        return np.asarray(vec) / numw



def doTheKM(data, model, evidences, filename):

    l=[]
    lr=[]
    for idx, i in enumerate(data):
        if len(i) > 0:
            vectorized = w2vectorizer(i, model)
            if len(vectorized) > 0:
                l.append(vectorized)
                lr.append(evidences[idx])
            #print(i)
    #for i in l:
    #    print(len(i))

    X=np.array(l)
    #print(X)
    #wcss=[]
    #for i in range(1,169):
    #    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    #    kmeans.fit(X)
    #    wcss.append(kmeans.inertia_)
    #plt.plot(range(1,169), wcss)
    #plt.title('The Elbow Method')
    #plt.xlabel('NUmber of clusters')
    #plt.ylabel('WCSS')
    #plt.show()

    n_clusters = 2
    clf = KMeans(n_clusters=n_clusters,
                 max_iter=1000,
                 init='k-means++',
                 n_init=1)
    labels = clf.fit_predict(X)

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

    f = open('./'+filename, "w")
    f.write("PCA on evidences")
    f.write("\n")
    f.write("----------------")
    f.write("\n")
    f.write("----------------")
    f.write("\n")
    f.write("\n")

    pca = PCA(n_components=2).fit(X)
    coords = pca.transform(X)
    label_colors = ['#2AB0E9', '#2BAF74', '#D7665E', '#CCCCCC',
                    '#D2CA0D', '#522A64', '#A3DB05', '#FC6514',
                    '#FF7F50', '#BDB76B', '#FF7F50', '#00FA9A',
                    '#FFA07A', '#FFFACD', '#006400', '#32CD32',
                    '#DC143C', '#FFEFD5', '#8FBC8F', '#808000'
                    ]

    #print(labels)
    colors = [label_colors[i] for i in labels]
    plt.scatter(coords[:, 0], coords[:, 1], c=colors)
    centroids = clf.cluster_centers_
    centroid_coords = pca.transform(centroids)
    plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=200,
    linewidths=2, c='#444d61')

    #print(pca.explained_variance_ratio_)
    #print(pca.components_)
    towrite=[]

    for r in range(n_clusters):
        f.write("Sentences in cluster n" + str(r) + ":")
        f.write("\n")
        f.write(".....................................")
        f.write("\n")
        cluster_cl = [lr[idx] for idx, x in enumerate(labels) if labels[idx]==r]
        for c in cluster_cl:
            f.write("\t" + c)
            f.write("\n")
            f.write("\n")

        f.write("\n")
        f.write("\n")
        f.write("\n")

    f.close()

    plt.show()

    return None
