from nltk.cluster import KMeansClusterer, euclidean_distance

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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

import mpld3
#import seaborn as sns




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
    #plt.show()


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

    ax.set_title("Scatter Plot (with tooltips!)", size=20)

    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    mpld3.plugins.connect(fig, tooltip)

    mpld3.show()


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
    #ax.tight_layout()
    #plt.show()


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

    fig, axes = plt.subplots(1, 2, figsize=(40,10))
    fig.suptitle('ScatterPlots of Methods')
    axes[0].set_title('Bigram Model', fontsize=14)
    axes[1].set_title('Trigram Model', fontsize=14)

    tsne_plot_custom(bigrams_model, men, axes[0])
    tsne_plot_custom(trigrams_model, men, axes[1])

    fig.tight_layout()
    #plt.show()


    mpld3.show()
    plt.savefig('TSNEbitri.png')

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
    plt.savefig('Scatterbitri.png')


def W2V_plot_Models_together(bi_vectors, tri_vectors, bi_labels, tri_labels):

    fig, axes = plt.subplots(1, figsize=(50,10))
    fig.suptitle('ScatterPlots of Methods')
    axes.set_title('BiTrigram Model', fontsize=14)
    #axes[1].set_title('Trigram Model', fontsize=14)

    scatter_Model(bi_vectors, bi_labels, axes)
    scatter_Model(tri_vectors, tri_labels, axes)

    fig.tight_layout()
    #plt.show()
    plt.savefig('SCatter_bitri_together.png')


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

    #plt.show()


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
        hull = ConvexHull(cluster_cl)
        #plt.plot(cluster_cl[:,0], cluster_cl[:,1], 'o')
        for simplex in hull.simplices:
            axs.plot(cluster_cl[simplex, 0], cluster_cl[simplex, 1], label_colors[idy] ,'k-')
            axs.fill(cluster_cl[hull.vertices,0], cluster_cl[hull.vertices,1], label_colors[idy], alpha=0.05)


def clusterPlot_Models(type, bigrams_, bigrams_model, trigrams_, trigrams_model, lemmatized):


    X, lr = vectorizeToX(bigrams_, bigrams_model, lemmatized)
    X2, lr2 = vectorizeToX(trigrams_, trigrams_model, lemmatized)

    plot_elbows(X, X2)

    print("After seeing the elbow method, insert the number of clusters")
    bi_clusters = int(input("Clusters for the Bigram Model: "))
    tri_clusters = int(input("Clusters for the Trigram Model: "))

    fig, axes = plt.subplots(1, figsize=(20,20))
    axes.set_title('Bigram Model', fontsize=14)
    #axes[1].set_title('Trigram Model', fontsize=14)

    axes.set(xlabel='Number of clusters', ylabel='WCSS')

    Kmeans_PCA(fig, axes, bi_clusters, X, lr, "bigram_"+type+".txt")
    plt.savefig(type+'_bigrams.png')

    fig, axes = plt.subplots(1, figsize=(20,20))
    axes.set_title('Bigram Model', fontsize=14)
    #axes[1].set_title('Trigram Model', fontsize=14)

    axes.set(xlabel='Number of clusters', ylabel='WCSS')

    Kmeans_PCA(fig, axes, tri_clusters, X2, lr2, "trigram_"+type+".txt")
    plt.savefig(type+'_trigrams.png')

    #plt.show()



##############################################


def fill_titles(full_content):

    template = full_content

    start_id = template.find('name="paper_titles">')+len('name="paper_titles">')
    #print(start_id)
    end_id = template[start_id:].find("</div>")
    start = template[:start_id]
    end = template[start_id+end_id:]

    fp = open('./paper_titles.txt', "r")
    file_orig = fp.read()
    file=file_orig
    message = "\n"
    count=0
    finished=False
    while finished==False:
        title_start = file.find("- ")
        if title_start != -1:
            end_title = file.find("---------------------")
            if len(file[title_start:end_title]) > 3:
                message = message + "\t\t\t\t\t\t\t<p> ("+ str(count+1) +") "+ (file[title_start:end_title].replace("\n\r", " ")) + "\t\t\t\t\t\t\t</p>" + "\n"
                count+=1
            if file[end_title:].find("\n") != -1:
                new_file_id = file[end_title:].find("\n")+1
                file = file[new_file_id:]
            else:
                finished = True
        else:
            finished = True

    fp.close()

    content = start+message+end
    return content


def fill_clustering(arg_name, full_content):

        if full_content == "":
            f = open('output_template.html','r')
            template = f.read()
            f.close()
        else:
            template=full_content

        start_id = template.find('name="'+arg_name+'">')+len('name="'+arg_name+'">')
        #print(start_id)
        end_id = template[start_id:].find("</div>")
        start = template[:start_id]
        end = template[start_id+end_id:]

        fp = open('./'+arg_name+'.txt', "r")
        line = fp.readline()
        message = "\n"
        count=0
        while line:
            line = fp.readline()
            #print(line)
            if line.find("---") != -1 or line.find("....") != -1:
                continue
            elif line.find("Sentences in cluster n") != -1:
                count+=1
                if count > 1:
                    message = message + "\t\t\t\t\t</div>" + "\n"

                message = message + '\t\t\t\t\t\t<button type="button" id="subsubcollapsible" class="collapsible">' + line[line.find("Sentences in cluster n"):line.find("Sentences in cluster n")+len("Sentences in cluster n")+1] + "</button>" + "\n"
                message = message + '\t\t\t\t\t\t<div class="content">' + "\n"

            else:
                message = message + "\t\t\t\t\t\t\t<p> "+ line + "\t\t\t\t\t\t\t</p>" + "\n"
                message = message + "\t\t\t\t\t\t\t<p>  \t\t\t\t\t\t\t</p>" + "\n"

        fp.close()
        message = message + "\t\t\t\t\t</div>" + "\n"

        content=start+message+end

        return content


def fill_mentions(full_content):

    template = full_content

    start_id = template.find('name="mentions">')+len('name="mentions">')
    #print(start_id)
    end_id = template[start_id:].find("</div>")
    start = template[:start_id]
    end = template[start_id+end_id:]

    fp = open('./pointed_mentions.txt', "r")

    line = fp.readline()
    message = "\n"
    count=0
    while line:
        line = fp.readline()

        if line.find("--------") != -1:
            continue
        #print(line)
        message = message + "\t\t\t\t\t\t\t<p> "+ line + "\t\t\t\t\t\t\t</p>" + "\n"
        message = message + "\t\t\t\t\t\t\t<p>  \t\t\t\t\t\t\t</p>" + "\n"

    fp.close()

    content = start+message+end
    return content


def dynamicfill_output():

    full_content = ""
    full_content = fill_clustering("bigram_claims", full_content)
    full_content = fill_clustering("trigram_claims", full_content)
    full_content = fill_clustering("bigram_evidences", full_content)
    full_content = fill_clustering("trigram_evidences", full_content)

    full_content = fill_titles(full_content)
    full_content = fill_mentions(full_content)

    f = open('./batch_statistics_view.html', "w")
    f.write(full_content)
    f.close()


#    message+message+msg
#    f.write(message)
    #f.close()

    return None
