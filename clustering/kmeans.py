from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import random
import json
import sys
from os import listdir
from os.path import isfile, join
import os


documents = ["This little kitty came to play when I was eating at a restaurant.",
             "Merley has the best squooshy kitten belly.",
             "Google Translate app is incredible.",
             "If you open 100 tab in google you get a smiley face.",
             "Best cat photo I've ever taken.",
             "Climbing ninja cat.",
             "Impressed with google map feedback.",
             "Key promoter extension for Google Chrome."]


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
        evidences.append(jso[idx]['evidence'])

    f.close()

import numpy as np

def term_scorer(doc_term_matrix, feature_name_list, labels=None, target=None, n_top_words=10):

    if target is not None:
        filter_bool = np.array(labels) == target
        doc_term_matrix = doc_term_matrix[filter_bool]
    term_scores = np.sum(doc_term_matrix,axis=0)
    top_term_indices = np.argsort(term_scores)[::-1]
    top_term_indices = np.squeeze(np.asarray(top_term_indices))

    return [feature_name_list[term_idx] for term_idx in top_term_indices[:n_top_words]]


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(evidences)

true_k = 4
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
km.fit(X)
labels = km.fit_predict(X)
X = X.todense()

print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

#print(term_scorer(X, terms, labels=model.labels_, target=2, n_top_words=15))

labels_color_map = {
    0: '#20b2aa', 1: '#ff7373', 2: '#ccc0ba', 3: '#005073', 4: '#4d0404',
    5: '#ffe4e1', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
}#ffe4e1
pca_num_components = 2
tsne_num_components = 2


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

tsne_init = 'pca'  # could also be 'random'
tsne_perplexity = 20.0
tsne_early_exaggeration = 4.0
tsne_learning_rate = 1000
random_state = 1
model = TSNE(n_components=2, random_state=random_state, init=tsne_init, perplexity=tsne_perplexity,
         early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate)

transformed_centroids = model.fit_transform(order_centroids)

reduced_data = PCA(n_components=pca_num_components).fit_transform(X)

fig, ax = plt.subplots()
#ax.scatter(order_centroids[:, 0], order_centroids[:, 1], marker='x')

for index, instance in enumerate(reduced_data):
    # print instance, index, labels[index]
    pca_comp_1, pca_comp_2 = reduced_data[index]
    color = labels_color_map[labels[index]]
    ax.scatter(pca_comp_1, pca_comp_2, c=color)
plt.show()


Sum_of_squared_distances = []
K = range(1,13)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#embeddings = TSNE(n_components=tsne_num_components)
#Y = embeddings.fit_transform(X)
#plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
#plt.show()

#plt.scatter(X[:, 0], X[:, 1], c='yellow', s=50, cmap='viridis')
#plt.scatter(order_centroids[:, 0], order_centroids[:, 1], c='black', s=200, alpha=0.5);
#plt.scatter(transformed_centroids[:, 0], transformed_centroids[:, 1], marker='x')
#plt.show()
