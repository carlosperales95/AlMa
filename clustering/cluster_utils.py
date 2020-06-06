from nltk.cluster import KMeansClusterer, euclidean_distance

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import sys
import os

from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
import numpy as np



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


def vectorizeToX(data, model, evidences):

    l=[]
    lr=[]
    for idx, i in enumerate(data):
        if len(i) > 0:
            vectorized = w2vectorizer(i, model)
            if len(vectorized) > 0:
                l.append(vectorized)
                lr.append(evidences[idx])

    X=np.array(l)

    return X, lr


def elbowMethod(X):

    wcss=[]
    for i in range(1,8):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,8), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


def plotKMtoPCA(X, labels, clf):

    pca = PCA(n_components=2).fit(X)
    coords = pca.transform(X)
    label_colors = ['#2AB0E9', '#2BAF74', '#D7665E', '#CCCCCC',
                    '#D2CA0D', '#522A64', '#A3DB05', '#FC6514',
                    '#FF7F50', '#BDB76B', '#FF7F50', '#00FA9A',
                    '#FFA07A', '#FFFACD', '#006400', '#32CD32',
                    '#DC143C', '#FFEFD5', '#8FBC8F', '#808000'
                    ]

    colors = [label_colors[i] for i in labels]
    plt.scatter(coords[:, 0], coords[:, 1], c=colors)
    centroids = clf.cluster_centers_
    centroid_coords = pca.transform(centroids)
    plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=200,
    linewidths=2, c='#444d61')
    plt.show()


def Kmeans_PCA(data, model, evidences, filename):

    X, lr = vectorizeToX(data, model, evidences)

    elbowMethod(X)

    n_clusters = int(input("After seeing the elbow method, select your number of clusters: "))
#    if n_clusters < 2:
#        n_clusters = 2

    clf = KMeans(n_clusters=n_clusters,
                 max_iter=1000,
                 init='k-means++',
                 n_init=1)
    labels = clf.fit_predict(X)


    f = open('./'+filename, "w")
    f.write("PCA on evidences")
    f.write("\n")
    f.write("----------------")
    f.write("\n")
    f.write("----------------")
    f.write("\n")
    f.write("\n")


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

    plotKMtoPCA(X, labels, clf)


    return None


def cluster_PlotAnnotate(clusters, vectors2, c_id, labels2):

    x0 = np.array([x[0] for idx, x in enumerate(vectors2) if clusters[idx]==c_id])
    y0 = np.array([x[1] for idx, x in enumerate(vectors2) if clusters[idx]==c_id])
    labels0 = np.array([labels2[idx] for idx, x in enumerate(vectors2) if clusters[idx]==c_id])
    plt.scatter(x0,y0, color='#FF7F50')
    for i, x in enumerate(x0):
        plt.annotate(labels0[i],
                     xy=(x0[i], y0[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    return None


def scatterW2V(vectors2, labels2):


    # test k-means using 2 means, euclidean distance, and 10 trial clustering repetitions with random seeds
    clusterer = KMeansClusterer(4, euclidean_distance, repeats=10)
    clusters = clusterer.cluster(vectors2, True)
    centroids = clusterer.means()
    print('Clustered ')
    print('As:', clusters)
    #print('Means:', centroids)


    for i in range(4):
        cluster_PlotAnnotate(clusters, vectors2, i, labels2)

    #x4 = np.array([x[0] for idx, x in enumerate(w2v_vectors) if clusters[idx]==4])
    #y4 = np.array([x[1] for idx, x in enumerate(w2v_vectors) if clusters[idx]==4])
    #plt.scatter(x4,y4, color='purple')


    #wv_vectors = model.wv.vectors
    # here you load indices - with whom you can find an index of the particular word in your model
    #wv_indices = {word: model.wv.vocab[word].index for word in model.wv.vocab}

    xc = np.array([x[0] for x in centroids])
    yc = np.array([x[1] for x in centroids])
    plt.scatter(xc,yc, color='red')
    plt.show()

    return None
