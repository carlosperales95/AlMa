import spacy
import gensim
import re
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


from word2utils import *
from htmlUtils import *
from cluster_utils import *

import warnings
import webbrowser

warnings.simplefilter("ignore")

#dir = "./rank/batch_503/"
dir = sys.argv[1]
nlp_base = spacy.load("en_core_web_sm")
nlp = spacy.load('mymodel')
lemmatizer = WordNetLemmatizer()


print("Joining Claims/Evidences....................")
evidences, claims = getClaimsEvidences(dir)
print("Number of Evidences: " + str(len(evidences)))
print("Number of Claims: " + str(len(claims)))
print("Total Number of Arguments: " + str(len(evidences) + len(claims)))
print("\n")


print("Lemmatizing Claims/Evidences....................")
print("\n")

lemmatized_sentences = []
lemmatized_c=[]
for c in claims:
    lemmatized_c.append(lemmatize_sentence(c[1], lemmatizer))
    lemmatized_sentences.append(lemmatize_sentence(c[1], lemmatizer))


lemmatized_e=[]
for e in evidences:
    lemmatized_e.append(lemmatize_sentence(e[1], lemmatizer))
    lemmatized_sentences.append(lemmatize_sentence(e[1], lemmatizer))



print("Converting Claims/Evidences to Bigrams and Trigrams....................")

bigrams_c, trigrams_c = convertSentsToBiTriGs(lemmatized_c)
#NEED TO CLEAN ARGS, BIGRAMS ARE TOO WEIRD
trigrams_c = filter_stringRubble(trigrams_c)

bigrams_e, trigrams_e = convertSentsToBiTriGs(lemmatized_e)
#NEED TO CLEAN ARGS, BIGRAMS ARE TOO WEIRD
trigrams_e = filter_stringRubble(trigrams_e)


#bigrams_, trigrams_ = convertSentsToBiTriGs(lemmatized_e)
bigrams_, trigrams_ = convertSentsToBiTriGs(lemmatized_sentences)
#NEED TO CLEAN ARGS, BIGRAMS ARE TOO WEIRD
trigrams_ = filter_stringRubble(trigrams_)

print("Found Bigrams on Claims: " + str(len(bigrams_c)))
print("Found Bigrams on Evidences: " + str(len(bigrams_e)))
print("Total Number of Bigrams: " + str(len(bigrams_)))
print("\n")
print("Found Trigrams on Claims: " + str(len(trigrams_c)))
print("Found Trigrams on Evidences: " + str(len(trigrams_e)))
print("Total Number of Trigrams: " + str(len(trigrams_)))
print("\n")



print("(word2vec) Creating Models....................")
print("\n")

STOP_WORDS = nltk.corpus.stopwords.words()

def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val)
    sentence = sentence.split(" ")

    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence


unigrams = []
for s in lemmatized_sentences:
    sentence = clean_sentence(s)
    sentence_tokens = nltk.word_tokenize(sentence)
    unigrams.append(sentence_tokens)

#print(unigrams)

unigrams_model = gensim.models.Word2Vec(
        unigrams,
        size=150,
        window=10,
        min_count=2,
        workers=10,
        iter=10)

bigrams_model = gensim.models.Word2Vec(
        bigrams_,
        size=150,
        window=10,
        min_count=2,
        workers=10,
        iter=10)

trigrams_model = gensim.models.Word2Vec(
        trigrams_,
        size=150,
        window=10,
        min_count=2,
        workers=10,
        iter=10)


#mentions = filterDates(mentions)
#mentions = filterWeirdChars(mentions)

print("Finding Technology/Method mentions....................")

pointed_mentions = getRankedMentions(lemmatized_sentences, nlp, nlp_base)

print("Number of potential mentions: " + str(len(pointed_mentions)))
###filterW2VSoft/HArd
#bi_vectors, bi_labels = W2V_filter_hard(bigrams_model, pointed_mentions)
#print("Mentions matching Bigrams: " + str(len(bi_labels)))

#tri_vectors, tri_labels = W2V_filter_soft(trigrams_model, pointed_mentions)
#print("Mentions matching Trigrams: " + str(len(tri_labels)))

    #W2V_plot_Models(bi_vectors, tri_vectors, bi_labels, tri_labels)
men = mentionRankThreshold(pointed_mentions)
#print("Number of mentions after score filter: " + str(len(men)))
print("\n")
#####fig, axes = plt.subplots(1, figsize=(20,20))
######axes.set_title('Bigram Model', fontsize=14)
#axes[1].set_title('Trigram Model', fontsize=14)

#axes.margins(1, 1)
plot_uni = tsne_plot_mentions(unigrams_model, men)
plot_bi = tsne_plot_mentions(bigrams_model, men)
plot_tri = tsne_plot_mentions(trigrams_model, men)

plots=[]
plots.append(plot_uni)
plots.append(plot_bi)
plots.append(plot_tri)

#####plt.show()
print("Plotting Bigrams/Trigrams models based on Technology/Method mentions....................")
print("\n")
#tsne_plot_Models(bigrams_model, trigrams_model, men)


print("Clustering/Plotting Evidence/Claim Models....................")

#clusterPlot_Models("claims", bigrams_c, bigrams_model, trigrams_c, trigrams_model, lemmatized_c, claims)
#clusterPlot_Models("evidences", bigrams_e, bigrams_model, trigrams_e, trigrams_model, lemmatized_e, evidences)


#######################################################################

print("Creating Output....................")

dynamicfill_output(men, claims, evidences, plots)
