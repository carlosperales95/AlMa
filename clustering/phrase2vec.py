import spacy
import gensim
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


from word2utils import *
from cluster_utils import *


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


bigrams_, trigrams_ = evid2bitriGrams(lemmatized_e)

#NEED TO CLEAN ARGS, BIGRAMS ARE TOO WEIRD
trigrams_ = filterStringRubble(trigrams_)


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


mentions = mentionsFromArgs(lemmatized_e, nlp, nlp_base)
mentions = filterWeirdChars(mentions)
mentions = filterStopwords(mentions)
mentions = filterSingleStrings(mentions)
true_mentions = filterDoubles(mentions)

semantic_mentions = getSemanticMentions(lemmatized_e)

true_mentions = filterSingleChars(true_mentions)
semantic_mentions = filterSingleChars(semantic_mentions)

pointed_mentions = addPoints(true_mentions, semantic_mentions, 1.3, 1.6)

pointed_mentions = sorted(pointed_mentions, key=lambda tup: tup[1], reverse=True)


f = open('./pointed_mentions.txt', "w")
f.write("List of scored Method/Technologies")
f.write("\n")
f.write("-----------------------------------")
f.write("\n")
f.write("\n")

for p in pointed_mentions:
    f.write(str(p))
    f.write("\n")

f.close()


print("Clustering/Plotting Bigrams/Trigrams....................")


#filterW2VSoft/HArd
bi_vectors, bi_labels = filterW2VHard(bigrams_model, pointed_mentions)
tri_vectors, tri_labels = filterW2VSoft(trigrams_model, pointed_mentions)

#w2v_vectors = model2.wv.vectors
# here you load indices - with whom you can find an index of the particular word in your model
#w2v_indices = {word: model2.wv.vocab[word].index for word in model2.wv.vocab}


scatterW2V(bi_vectors, bi_labels)
scatterW2V(tri_vectors, tri_labels)

#print("Clustering/Plotting Trigrams....................")


Kmeans_PCA(bigrams_, bigrams_model, lemmatized_e, "bigram_claims.txt")
Kmeans_PCA(trigrams_, trigrams_model, lemmatized_e, "trigram_claims.txt")



#tsne_plot(model)
#tsne_plot2(model, tris)
