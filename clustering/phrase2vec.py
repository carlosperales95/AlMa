import spacy
import gensim
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


from word2utils import *
from cluster_utils import *

import warnings
#warnings.simplefilter("ignore")

dir = "./rank/batch_503/"
nlp_base = spacy.load("en_core_web_sm")
nlp = spacy.load('mymodel')


print("Joining evidences....................")

evidences, claims = getClaimsEvidences(dir)

lemmatizer = WordNetLemmatizer()


lemmatized_e=[]
for e in evidences:
    lemmatized_e.append(lemmatize_sentence(e, lemmatizer))


lemmatized_c=[]
for c in claims:
    lemmatized_c.append(lemmatize_sentence(c, lemmatizer))


bigrams_, trigrams_ = convertSentsToBiTriGs(lemmatized_e)

#NEED TO CLEAN ARGS, BIGRAMS ARE TOO WEIRD
trigrams_ = filter_stringRubble(trigrams_)


print("(word2vec) Finding for bigrams and trigrams....................")


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

pointed_mentions = getRankedMentions(lemmatized_e, nlp, nlp_base)

print("Creating ScatterPlots of Bigrams/Trigrams....................")


#filterW2VSoft/HArd
bi_vectors, bi_labels = W2V_filter_hard(bigrams_model, pointed_mentions)
tri_vectors, tri_labels = W2V_filter_soft(trigrams_model, pointed_mentions)

#w2v_vectors = model2.wv.vectors
# here you load indices - with whom you can find an index of the particular word in your model
#w2v_indices = {word: model2.wv.vocab[word].index for word in model2.wv.vocab}


W2V_plot_Models(bi_vectors, tri_vectors, bi_labels, tri_labels)
men = mentionRankThreshold(pointed_mentions)
#men = [x[0] for x in men]
tsne_plot_custom(bigrams_model, men)
print("Clustering/Plotting Evidence Models....................")


#clusterPlot_Models(bigrams_, bigrams_model, trigrams_, trigrams_model, lemmatized_e)

#######################################################################





#tsne_plot(model)
#tsne_plot2(model, tris)
