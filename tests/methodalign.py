
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import json
import sys
from os import listdir
from os.path import isfile, join
import os

nlp_base = spacy.load("en_core_web_sm")
nlp = spacy.load('mymodel')

evidenci = ["we developed a graph alignment algorithm that iteratively reinforces the matching similarity",
            "We have shown that our cache-based approach significantly improves the performance with the help of various caches",
            "We can construct a topic model once on the training data , and use it infer topics on any test set to adapt the translation model .",
            "We induce unsupervised domains from large corpora , and we incorporate soft , probabilistic domain membership into a translation model .",
            "We believe that by performing a rescoring on translation word graphs we will obtain a more significant improvement in translation quality .",
            "We have been able to obtain a significant better test corpus perplexity and also a slight improvement in translation quality ."]


dir = "./rank/batch_503/"


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



mentions = []

for idt, evidence in enumerate(evidences):

    text = word_tokenize(evidences[idt])
    tokens = nltk.pos_tag(text)

    tags = [lis[1] for lis in tokens]
    texts = [lis[0] for lis in tokens]

    phrases = []
    continuation = 0


    for idx, tag in enumerate(tags):
        subject = False
        verbal = False
        index = []
        index3 = []
        skiptomethod = False
        methodstart = False
        index2 = []
        other = False
        proper_method = False
        methodstart_id = 0

        if continuation > 0:
            idx = continuation
        #print(idx)
        if tags[idx] == 'PRP':
            #print("PRP: " + texts[idx])
            subject = True
            index.append(idx)

        elif tags[idx] == 'DT' and tags[idx+1].startswith('NN'):
            #print("(DT)+NN: " + texts[idx] + " " + texts[idx+1])
            subject = True
            index.append(idx)
            index.append(idx+1)

        elif tags[idx].startswith('NN'):
            #print("NN: " + texts[idx])
            subject = True
            index.append(idx)


        end = len(index)
        if subject:
            for idy, t in enumerate(tags):
                #print("idx: " + str(idx) + " / " + str(index[end-1]))
                if idy > index[end-1]:
                    #print("PASS")
                    if t.startswith('VB'):
                        #print(tags[idy] + ': ' + texts[idy])
                        verbal = True
                        index2.append(idy)
                        id = idy
                        tagloop = tags[id+1:]
                        for idz, tl in enumerate(tagloop):
                            while not other:
                                if tagloop[idz].startswith('VB'):
                                    #print(tagloop[idz] + ': ' + texts[id + (idz + 1)])
                                    index2.append(id + (idz + 1))
                                    idz += 1
                                else:
                                    other = True
                            break
                        if verbal and other:
                            break
                    else:
                        verbal = False
                        break
            end2 = len(index2)
            other = False
            if subject and verbal:
                for idz, tt in enumerate(tags):
                    if idz > index2[end2-1]:
                        if tt.startswith('JJ'):
                            methodstart = True
                            #print("JJ: " + texts[idz])
                            index3.append(idz)
                            break
                        elif len(tags) > (idz + 1):
                            if tt == 'DT' and ( tags[idz+1].startswith('NN') or tags[idz+1].startswith('JJ')):
                                #print("(DT)+NN: " + texts[idz] + " " + texts[idz+1])
                                methodstart = True
                                index3.append(idz)
                                index3.append(idz+1)
                                break
                            elif tt not in ['PRP$', 'IN']:
                                methodstart = False
                                break

                if methodstart:
                    #print(index3[len(index3) - 1])
                    id = index3[len(index3) - 1]
                    methodstart_id = index3[len(index3) - 1]
                    tagloop = tags[id+1:]
                    #print(id)
                    #print(tagloop)
                    for idw, tl in enumerate(tagloop):
                        while not other:
                            #print(id+(idw+1))
                            #print(len(texts))
                            #print(texts[id+(idw+1)])
                            #print("This one is: " + tagloop[idw] + " / " + texts[id+(idw+1)] + " |vs: " + tl)
                            if tagloop[idw].startswith('NN'):
                                proper_method = True

                            if tagloop[idw].startswith('NN') or tagloop[idw].startswith('JJ'):
                                #print(len(texts))
                                #print(str(id + (idw)))
                                #print(tagloop[idw] + ': ' + texts[id + (idw+1)])
                                index3.append(id + (idw+1))
                                if (id+idw+1) == (len(texts)-1):
                                    other = True
                                    break
                                else:
                                    idw += 1
                            else:
                                other = True
                                break
                        #break


                    #print("\n")
                                #print("\n")
                                #print("\n")
        if proper_method:
            phrase = ""
            mention = []
            for ind in index:
                phrase = phrase + "(" + tags[ind]  + ") " + texts[ind] + " "
            for ind in index2:
                phrase = phrase + "(" + tags[ind]  + ") " + texts[ind] + " "
            for ind in index3:
                phrase = phrase + "(" + tags[ind]  + ") " + texts[ind] + " "
                if(ind >= methodstart_id):
                    mention.append(texts[ind])
            phrases.append(phrase)
            mentions.append(mention)

            continuation = index3[(len(index3) - 1)] + 1



    for phrase in phrases:
        print(phrase)
        print("\n")
for mention in mentions:
    print(mention)



#    for idx, tag in enumerate(tags):
#        phrase = ""
#        phrase = phrase + "[" + tag + "], " + text[idx] + "/"
        #print(phrase)
    #print("\n")
