from nltk.cluster import KMeansClusterer, euclidean_distance

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import string
import spacy
import json


import sys
import os

from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


from os import listdir
from os.path import isfile, join

import mpld3

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
    #print(mentions)
    for lab in label:
        for mention in mentions:
            if isinstance(mention, list):
                if lab in mention:
                    count+=1
                    #print(lab + " matched")
                    continue
                if count == len(label):
                    important = True
                    break
            elif lab == mention:
                important = True
                break

    return important


def tsne_plot_mentions(model, mentions):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    nlp_base = spacy.load("en_core_web_sm")


    for word in model.wv.vocab:
        if isImportant(word, mentions) == True:
            label = word.split(' ')
            no_stopword_word = ""

            for la in label:
                if la in set(stopwords.words('english')):
                    continue
                if la in string.punctuation:
                    continue
                else:
                    no_stopword_word = no_stopword_word + " " + la

            doc = nlp_base(no_stopword_word)
            ents = doc.ents
            ignore = False
            if len(ents)>0:
                for e in ents:
                    if hasattr(e, 'label_') and e.label_ == 'PERSON':
                    #    #print(e.label_ + " | " + e.text)
                        ignore = True
                        break
            if ignore == False:
                double_c = no_stopword_word.split(' ')
                if len(double_c) > 1:
                    labels.append(no_stopword_word)
                    tokens.append(model[word])

    #print(labels)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'))

    scatter = ax.scatter(x,
                         y,
                         cmap=plt.cm.jet)
    ax.grid(color='white', linestyle='solid')

    #ax.set_title("Model", size=20)

    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    mpld3.plugins.connect(fig, tooltip)

    #mpld3.show()
    html_str = mpld3.fig_to_html(fig, template_type='simple')

    return html_str


############## MAINLY USED TSNE #############


def tsne_plot_custom(model, mentions, ax):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    nlp_base = spacy.load("en_core_web_sm")


    for word in model.wv.vocab:
        if isImportant(word, mentions) == True:
            label = word.split(' ')
            no_stopword_word = ""

            for la in label:
                if la in set(stopwords.words('english')):
                    continue
                if la in string.punctuation:
                    continue
                else:
                    no_stopword_word = no_stopword_word + " " + la

            doc = nlp_base(no_stopword_word)
            ents = doc.ents
            ignore = False
            if len(ents)>0:
                for e in ents:
                    if hasattr(e, 'label_') and e.label_ == 'PERSON':
                    #    #print(e.label_ + " | " + e.text)
                        ignore = True
                        break
            if ignore == False:
                double_c = no_stopword_word.split(' ')
                if len(double_c) > 1:
                    labels.append(no_stopword_word)
                    tokens.append(model[word])
                    #print(no_stopword_word)


    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    #ax.figure(figsize=(50, 20))
    for i in range(len(x)):
        ax.scatter(x[i],y[i])
        ax.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')


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


def vectorizeToX(data, model, lemmatized):

    l=[]
    lr=[]
    for idx, i in enumerate(data):
        if len(i) > 0:
            vectorized = w2vectorizer(i, model)
            if len(vectorized) > 0:
                l.append(vectorized)
                lr.append(lemmatized[idx])

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


def Kmeans_PCA(sentences, fig, axs, n_clusters, X, lr, filename):

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
        cluster_cl = [sentences[idx] for idx, x in enumerate(labels) if labels[idx]==r]
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


def scatter_Model(vectors, labels, ax):

    new_vecs = []
    new_labs = []
    for idx, lab in enumerate(labels):
        lab_splt = lab.split(' ')
        no_stopword_lab = ""
        if isinstance(lab_splt, list):
            for la in lab_splt:
                if la in set(stopwords.words('english')):
                    continue
                if la in string.punctuation:
                    continue
                else:
                    no_stopword_lab = no_stopword_lab + " " + la

        new_labs.append(no_stopword_lab)
        new_vecs.append(vectors[idx])


    # test k-means using 2 means, euclidean distance, and 10 trial clustering repetitions with random seeds
    clusterer = KMeansClusterer(4, euclidean_distance, repeats=10)
    clusters = clusterer.cluster(new_vecs, True)
    centroids = clusterer.means()
    print('Clustered ')
    print('As:', clusters)
    #print('Means:', centroids)


    for i in range(4):
        scatter_Annotate(clusters, new_vecs, i, new_labs, ax)

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


def tsne_plot_Models(bigrams_model, trigrams_model, men):

    #fig.suptitle('ScatterPlots of Methods')
    #fig.tight_layout()
    #plt.show()
    fig, axes = plt.subplots(1, figsize=(20,20))
    axes.set_title('Bigram Based', fontsize=14)

    tsne_plot_custom(bigrams_model, men, axes)
    plt.tight_layout()

    plt.savefig('./outs/men_bigrams.png')
    plt.close()

    fig, axes = plt.subplots(1, figsize=(20,20))
    axes.set_title('Trigram Based', fontsize=14)
    tsne_plot_custom(trigrams_model, men, axes)

    plt.tight_layout()
    plt.savefig('./outs/men_trigrams.png')
    plt.close()
    return None



def W2V_plot_Models(bi_vectors, tri_vectors, bi_labels, tri_labels):

    fig, axes = plt.subplots(1, 2, figsize=(40,10))
    fig.suptitle('ScatterPlots of Methods')
    axes[0].set_title('Bigram Model', fontsize=14)
    axes[1].set_title('Trigram Model', fontsize=14)

    scatter_Model(bi_vectors, bi_labels, axes[0])
    scatter_Model(tri_vectors, tri_labels, axes[1])

    fig.tight_layout()
    #plt.show()
    plt.savefig('./outs/Scatterbitri.png')
    plt.close()


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
    label_colors = ['#2AB0E9', '#D2CA0D', '#D7665E', '#2BAF74',
                    '#CCCCCC', '#522A64', '#A3DB05', '#FC6514',
                    '#FF7F50', '#BDB76B', '#FF7F50', '#00FA9A',
                    '#FFA07A', '#FFFACD', '#006400', '#32CD32',
                    '#DC143C', '#FFEFD5', '#8FBC8F', '#808000'
                    ]
    colors = [label_colors[i] for i in labels]
    axs.scatter(coords[:, 0], coords[:, 1], c=colors)

    centroids = clf.cluster_centers_
    #centroid_coords = pca.transform(centroids)
    #axs.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=10,
    #linewidths=2, c='#444d61')

    for idy,r in enumerate(centroids):
        cluster_cl = np.array([coords[idx] for idx, x in enumerate(labels) if labels[idx]==idy])
        #cluster_cl=np.asarray(cluster_cl)
        if len(cluster_cl > 2):
            hull = ConvexHull(cluster_cl)
            #plt.plot(cluster_cl[:,0], cluster_cl[:,1], 'o')
            for simplex in hull.simplices:
                axs.plot(cluster_cl[simplex, 0], cluster_cl[simplex, 1], label_colors[idy] ,'k-')
                axs.fill(cluster_cl[hull.vertices,0], cluster_cl[hull.vertices,1], label_colors[idy], alpha=0.05)


def clusterPlot_Models(type, bigrams_, bigrams_model, trigrams_, trigrams_model, lemmatized, sentences):


    X, lr = vectorizeToX(bigrams_, bigrams_model, lemmatized)
    X2, lr2 = vectorizeToX(trigrams_, trigrams_model, lemmatized)

    print("Plotting elbow Method for cluster input for Bigram and Trigram models in "+type+":")
    plot_elbows(X, X2)

    print("After seeing the elbow method, insert the number of clusters")
    bi_clusters = int(input("Clusters for the "+ type +" Bigram Model: "))
    tri_clusters = int(input("Clusters for the "+ type +" Trigram Model: "))
    print("\n")

    fig, axes = plt.subplots(1, figsize=(20,20))
    axes.set_title('Bigram Model', fontsize=14)
    #axes[1].set_title('Trigram Model', fontsize=14)

    axes.set(xlabel='Number of clusters', ylabel='WCSS')

    Kmeans_PCA(sentences, fig, axes, bi_clusters, X, lr, "./outs/bigram_"+type+".txt")
    plt.savefig('./outs/'+type+'_bigrams.png')
    plt.close()

    fig, axes = plt.subplots(1, figsize=(20,20))
    axes.set_title('Bigram Model', fontsize=14)
    #axes[1].set_title('Trigram Model', fontsize=14)

    axes.set(xlabel='Number of clusters', ylabel='WCSS')

    Kmeans_PCA(sentences, fig, axes, tri_clusters, X2, lr2, "./outs/trigram_"+type+".txt")
    plt.savefig('./outs/'+type+'_trigrams.png')
    plt.close()











######################      WIP       #######################3


def W2V_plot_Models_together(bi_vectors, tri_vectors, bi_labels, tri_labels):

    fig, axes = plt.subplots(1, figsize=(50,10))
    fig.suptitle('ScatterPlots of Methods')
    axes.set_title('BiTrigram Model', fontsize=14)
    #axes[1].set_title('Trigram Model', fontsize=14)

    scatter_Model(bi_vectors, bi_labels, axes)
    scatter_Model(tri_vectors, tri_labels, axes)

    fig.tight_layout()
    #plt.show()
    plt.savefig('./outs/SCatter_bitri_together.png')
    plt.close()



def tsne_plot_custompap(found_m, model, mentions, ax):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    nlp_base = spacy.load("en_core_web_sm")


    for word in model.wv.vocab:
        if isImportant(word, mentions) == True:
            label = word.split(' ')
            no_stopword_word = ""

            for la in label:
                if la in set(stopwords.words('english')):
                    continue
                if la in string.punctuation:
                    continue
                else:
                    no_stopword_word = no_stopword_word + " " + la

            doc = nlp_base(no_stopword_word)
            ents = doc.ents
            ignore = False
            if len(ents)>0:
                for e in ents:
                    if hasattr(e, 'label_') and e.label_ == 'PERSON':
                    #    #print(e.label_ + " | " + e.text)
                        ignore = True
                        break
            if ignore == False:
                double_c = no_stopword_word.split(' ')
                if len(double_c) > 1:
                    labels.append(no_stopword_word)
                    tokens.append(model[word])


    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)



    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    #ax.figure(figsize=(50, 20))
    for i in range(len(x)):
        ax.scatter(x[i],y[i])
        ax.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    hulls=[]
    for found in found_m:
        papershape=[]
        for idx, flab in enumerate(found):
            #print(flab)
            for idy, lab in enumerate(labels):
                #print(flab[0:])
                if flab[1:] == lab:
                    print("FOUND IT BOSS")
                    papershape.append(new_values[idy])
                    break
    print(len(found_m))
    print(len(hulls))
    for idy, r in enumerate(hulls):
        hull_vals = np.array([r])
        #cluster_cl=np.asarray(cluster_cl)
        ##x = []
        ##y = []
        ##for value in hull_vals:
            ##x.append(value[0])
            ##y.append(value[1])

        hull = ConvexHull(hull_vals)
        #plt.plot(cluster_cl[:,0], cluster_cl[:,1], 'o')
        for simplex in hull.simplices:
            axs.plot(hull_vals[simplex, 0], hull_vals[simplex, 1], '#A3DB05' ,'k-')
            axs.fill(hull_vals[hull.vertices,0], hull_vals[hull.vertices,1], '#A3DB05' , alpha=0.05)



def tsne_plot_ModelsPapers(bigrams_model,  men):

    #fig.suptitle('ScatterPlots of Methods')
    #fig.tight_layout()
    #plt.show()

    dir = "./sum_batch/"

    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

    fig, axes = plt.subplots(1, figsize=(20,20))
    axes.set_title('Bigram Based', fontsize=14)

    texts=[]
    for f in onlyfiles:
        file = open(dir+f, 'r')
        t = file.read()
        texts.append(t)

    all_f_m=[]
    for idx, paper in enumerate(texts):
        found_men=[]

        for m in men:
            f_m=""
            if isinstance(m, list):
                for idy, p in enumerate(m):
                    if idy == 0:
                        f_m = p
                    else:
                        f_m = f_m + " " + p
            else:
                f_m = m
            if paper.find(f_m) != -1:
                found_men.append(f_m)

        all_f_m.append(found_men)



    tsne_plot_custompap(all_f_m, bigrams_model, men, axes)
    plt.tight_layout()

    plt.savefig('./outs/NewMen_bigrams.png')
    plt.close()

    #mpld3.show()

    return None
