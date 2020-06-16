from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from os import listdir
from os.path import isfile, join
import os
from word2utils import *
import spacy
from nltk.stem import WordNetLemmatizer




dir= str(sys.argv[1])

dir = "./rank/batch_503/"
nlp_base = spacy.load("en_core_web_sm")
nlp = spacy.load('mymodel')


print("Joining evidences....................")

evidences, claims = getClaimsEvidences(dir)
lemmatizer = WordNetLemmatizer()

sentences = claims
lemmatized_s=[]
for e in evidences:
    lemmatized_s.append(lemmatize_sentence(e, lemmatizer))
    sentences.append(e)

for c in claims:
    lemmatized_s.append(lemmatize_sentence(c, lemmatizer))


bigrams_, trigrams_ = convertSentsToBiTriGs(lemmatized_s)

#NEED TO CLEAN ARGS, BIGRAMS ARE TOO WEIRD
trigrams_ = filter_stringRubble(trigrams_)


#mentions = filterDates(mentions)
#mentions = filterWeirdChars(mentions)

print("Finding Technology/Method mentions....................")

pointed_mentions = getRankedMentions(lemmatized_s, nlp, nlp_base)


men = mentionRankThreshold(pointed_mentions)

#print(men[0])


def crossRef(evid, men):

    definit=[]
    for ev in evid:
        evi=ev.split(" ")
        for word in evi:
            important=False
            if len(word)<3:
                continue
            for m in men:
                if isinstance(m, list):
                    ml = [each_string.lower() for each_string in m]
                    if word.lower() in ml:
                        important = True
                        break
                elif word.lower()==m.lower():
                    important = True
                    break
            if important == False:
                definit.append(word)


    return definit



# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import seaborn as sns
sns.set_style('whitegrid')

def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()# Initialise the count vectorizer with the English stop words


others = crossRef(sentences, men)
others.append("et")
others.append("al")
my_stop_words = text.ENGLISH_STOP_WORDS.union(others)
print(others)
#print(my_stop_words)

count_vectorizer = CountVectorizer(stop_words=my_stop_words)# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(sentences)# Visualise the 10 most common words
#count_data = crossRef(count_data, men)
#plot_10_most_common_words(count_data, count_vectorizer)


import warnings
warnings.simplefilter("ignore", DeprecationWarning)# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Helper function
def print_topics(model, count_vectorizer, n_top_words, men):
    words = count_vectorizer.get_feature_names()

    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

# Tweak the two parameters below
number_topics = 5
number_words = 10# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)# Print the topics found by the LDA model
print("Topics found via LDA:")
#print_topics(lda, count_vectorizer, number_words, men)

import pyLDAvis
from pyLDAvis import sklearn as sklearn_lda
import pickle

LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(number_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
    f.close()
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    #f.encode('utf-8').strip()
    LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_'+ str(number_topics) +'.html')
