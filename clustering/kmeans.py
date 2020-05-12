from sklearn.cluster import KMeans
import numpy as np
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


dir= str(sys.argv[1])
onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
evidences = []
claims = []
paper_evidences = []
paper_claims = []


for file in onlyfiles:

    if file.startswith( 'evidence' ):
        paper_evidences.append(file)

    if file.startswith( 'claim' ):
        paper_claims.append(file)

for idx,file in enumerate(paper_evidences):

    f = open(dir+file, "r")
    paper_full =f.read()

    jso = json.loads(paper_full)
    for idx,item in enumerate(jso):
        evidences.append(jso[idx]['evidence'])

    f.close()


x = np.random.random(len(evidences))

km = KMeans()
km.fit(x.reshape(-1,1))
