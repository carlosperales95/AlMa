import nltk
from nltk.tokenize import word_tokenize
import spacy
import random
import json
import sys
from os import listdir
from os.path import isfile, join
import os



dir = "./rank/batch_2178/"


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

nlp_base = spacy.load("en_core_web_sm")
nlp = spacy.load('mymodel')

f = open("evidencesPOS.txt", "w")

for idx, evidence in enumerate(evidences):

    text = word_tokenize(evidences[idx])
    tokens = nltk.pos_tag(text)

    tags = [lis[1] for lis in tokens]
    texts = [lis[0] for lis in tokens]

    precandidates = []
    candidates = []
    doc = nlp(evidence)
    entities = doc.ents
    doc_base = nlp_base(evidence)
    entities_base = doc_base.ents
    entities_final = []
    for ent in entities_base:
      for dent in entities:
          if hasattr(ent, 'label_') and hasattr(dent, 'label_'):
              if ent.text.find(dent.text):
                  if ent.label_ == 'PRODUCT' or ent.label_ == 'ORG' or ent.label_ == 'NORP':
                      entities_final.append(dent)
                  #else:
              else:
                  entities_final.append(dent)

    for idx, tag in enumerate(tags):
        if tag == 'NN' or tag.startswith('JJ'):
            #print(idx)
            precandidates.append(texts[idx])
            candidates.append(texts[idx+1])

    mentions = []
    for candidate in candidates:
        for ent in entities:
            if hasattr(ent, 'label_'):
                if ent.text.find(candidate):
                    mentions.append(ent)


    mentions = list(dict.fromkeys(mentions))

    f.write(evidence)
    f.write("\n")
    f.write(str(tokens))
    f.write("\n")

    for mention in mentions:
        f.write(mention.label_ + "|" + mention.text)
        f.write("\n")

    f.write("\n")
    f.write("\n")




f.close()
#Rule for method mention
# (Adjective | Noun)+(method | analysis | algorithm | approach| model)
