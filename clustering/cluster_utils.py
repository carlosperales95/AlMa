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

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull




################# TSNE ######################


def tsne_plot_original(model):

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
    count=0
    for lab in label:
        for mention in mentions:
            if lab in mention:
                count+=1
            if count == len(label):
                important = True
                break
    return important


def tsne_plot_mentions(model, mentions):
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


def tsne_plot_custom(model, mentions):
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

    plt.figure(figsize=(50, 20))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.tight_layout()
    plt.show()


##############################################

############### PLOT ADDITIONS ###############


def encircle2(axs, x,y, ax=None, **kw):
    if not ax:
        ax=fig.gca()
    p = np.c_[x,y]
    mean = np.mean(p, axis=0)
    d = p-mean
    r = np.max(np.sqrt(d[:,0]**2+d[:,1]**2 ))
    circ = plt.Circle(mean, radius=1.05*r,**kw)
    axs.add_patch(circ)

    return None


def encircle(fig, x,y, ax=None, **kw):
    if not ax:
        ax=fig.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)

    return None


##############################################

############### VECTORIZERS ##################


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


##############################################

############### CLUSTER METHODS ##############


def elbowMethod(X):

    wcss=[]
    for i in range(1,8):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    return wcss


def Kmeans_PCA(fig, axs, n_clusters, X, lr, filename):

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

    plot_PCA(fig, X, labels, clf, axs)


    return None


##############################################

################### PLOTS ####################


def scatter_Annotate(clusters, vectors2, c_id, labels2, ax):

    x0 = np.array([x[0] for idx, x in enumerate(vectors2) if clusters[idx]==c_id])
    y0 = np.array([x[1] for idx, x in enumerate(vectors2) if clusters[idx]==c_id])
    labels0 = np.array([labels2[idx] for idx, x in enumerate(vectors2) if clusters[idx]==c_id])
    ax.scatter(x0,y0, color='#FF7F50')
    for i, x in enumerate(x0):
        ax.annotate(labels0[i],
                     xy=(x0[i], y0[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    return None


def scatter_Model(vectors2, labels2, ax):


    # test k-means using 2 means, euclidean distance, and 10 trial clustering repetitions with random seeds
    clusterer = KMeansClusterer(4, euclidean_distance, repeats=10)
    clusters = clusterer.cluster(vectors2, True)
    centroids = clusterer.means()
    print('Clustered ')
    print('As:', clusters)
    #print('Means:', centroids)


    for i in range(4):
        scatter_Annotate(clusters, vectors2, i, labels2, ax)

    #x4 = np.array([x[0] for idx, x in enumerate(w2v_vectors) if clusters[idx]==4])
    #y4 = np.array([x[1] for idx, x in enumerate(w2v_vectors) if clusters[idx]==4])
    #plt.scatter(x4,y4, color='purple')


    #wv_vectors = model.wv.vectors
    # here you load indices - with whom you can find an index of the particular word in your model
    #wv_indices = {word: model.wv.vocab[word].index for word in model.wv.vocab}

    xc = np.array([x[0] for x in centroids])
    yc = np.array([x[1] for x in centroids])
    ax.scatter(xc,yc, color='red')
    #plt.show()

    return None


def W2V_plot_Models(bi_vectors, tri_vectors, bi_labels, tri_labels):

    fig, axes = plt.subplots(1, 2, figsize=(40,10))
    fig.suptitle('ScatterPlots of Methods')
    axes[0].set_title('Bigram Model', fontsize=14)
    axes[1].set_title('Trigram Model', fontsize=14)

    scatter_Model(bi_vectors, bi_labels, axes[0])
    scatter_Model(tri_vectors, tri_labels, axes[1])

    fig.tight_layout()
    plt.show()


def plot_elbows(X, X2):

    fig, axes = plt.subplots(1, 2, figsize=(20,20))
    fig.suptitle('Elbow Method for Evidences')
    axes[0].set_title('Bigram Model', fontsize=14)
    axes[1].set_title('Trigram Model', fontsize=14)

    for ax in axes.flat:
        ax.set(xlabel='Number of clusters', ylabel='WCSS')

    wcss1 = elbowMethod(X)
    wcss2 = elbowMethod(X2)

    axes[0].plot(range(1,8), wcss1)
    axes[1].plot(range(1,8), wcss2)

    plt.show()


def plot_PCA(fig, X, labels, clf, axs):

    pca = PCA(n_components=2).fit(X)
    coords = pca.transform(X)
    label_colors = ['#2AB0E9', '#2BAF74', '#D7665E', '#CCCCCC',
                    '#D2CA0D', '#522A64', '#A3DB05', '#FC6514',
                    '#FF7F50', '#BDB76B', '#FF7F50', '#00FA9A',
                    '#FFA07A', '#FFFACD', '#006400', '#32CD32',
                    '#DC143C', '#FFEFD5', '#8FBC8F', '#808000'
                    ]

    colors = [label_colors[i] for i in labels]
    axs.scatter(coords[:, 0], coords[:, 1], c=colors)
    centroids = clf.cluster_centers_
    centroid_coords = pca.transform(centroids)
    axs.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=10,
    linewidths=2, c='#444d61')


def clusterPlot_Models(bigrams_, bigrams_model, trigrams_, trigrams_model, lemmatized):


    X, lr = vectorizeToX(bigrams_, bigrams_model, lemmatized)
    X2, lr2 = vectorizeToX(trigrams_, trigrams_model, lemmatized)

    #plot_elbows(X, X2)

    print("After seeing the elbow method, insert the number of clusters")
    bi_clusters = int(input("Clusters for the Bigram Model: "))
    tri_clusters = int(input("Clusters for the Trigram Model: "))


    fig, axes = plt.subplots(1, 2, figsize=(20,20))
    fig.suptitle('Clustered Evidences (K-means)')
    axes[0].set_title('Bigram Model', fontsize=14)
    axes[1].set_title('Trigram Model', fontsize=14)

    for ax in axes.flat:
        ax.set(xlabel='Number of clusters', ylabel='WCSS')


    Kmeans_PCA(fig, axes[0], bi_clusters, X, lr, "bigram_evidences.txt")
    Kmeans_PCA(fig, axes[1], tri_clusters, X2, lr2, "trigram_evidences.txt")

    plt.show()


##############################################
