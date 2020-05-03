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



dir= str(sys.argv[1])
onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
evidences = []
sentences = []


for file in onlyfiles:

    if file.startswith( 'evidence' ):
        evidences.append(file)

for idx,file in enumerate(evidences):

    f = open(dir+file, "r")
    paper_full =f.read()

    jso = json.loads(paper_full)
    for idx,item in enumerate(jso):
        sentences.append(jso[idx]['evidence'])

    f.close()



for s in sentences:
    print(s)
    print("\n")


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(sentences)

print(X)
plt.scatter(X[:,0], X[:,1])

true_k = 3
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print


print("\n")
print("Prediction")

Y = vectorizer.transform(["We believe that our results on the computational complexity of the tasks in SMT will result in a better understanding of these tasks from a theoretical perspective."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["We evaluated the adapted LM on SMT and found that the evaluation metrics are crucial to reflect the actual improvement in performance."])
prediction = model.predict(Y)
print(prediction)
