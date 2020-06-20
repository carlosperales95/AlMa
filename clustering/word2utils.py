import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from gensim.models import Phrases

import re

import json
import sys
from os import listdir
from os.path import isfile, join

import numpy as np



def vectorize(line):

    words = []
    for word in line: # line - iterable, for example list of tokens

        try:
            w2v_idx = w2v_indices[word]
        except KeyError: # if you does not have a vector for this word in your w2v model, continue
            continue
        words.append(w2v_vectors[w2v_idx])
        if words:

            words = np.asarray(words)
            min_vec = words.min(axis=0)
            max_vec = words.max(axis=0)
            return np.concatenate((min_vec, max_vec))
        if not words:

            return None



############## FETCH SENTENCES ##############


def getClaimsEvidences(dir):

    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

    paper_evidences = []
    paper_claims = []
    for file in onlyfiles:

        if file.startswith( 'evidence' ):
            paper_evidences.append(file)

        if file.startswith( 'claim' ):
            paper_claims.append(file)


    evidences = []
    claims = []
    for idx,file in enumerate(paper_evidences):

        f = open(dir+file, "r")
        paper_full =f.read()

        jso = json.loads(paper_full)
        for idx,item in enumerate(jso):
            if jso[idx] != "":
                evidences.append(jso[idx]['text'])

        f.close()

    for idx,file in enumerate(paper_claims):

        f = open(dir+file, "r")
        paper_full =f.read()

        jso = json.loads(paper_full)
        for idx,item in enumerate(jso):
            if jso[idx] != "":
                claims.append(jso[idx]['text'])

        f.close()


    return evidences, claims


##############################################

############# SEMATINC FEATS #################


def convert_nltkToWN_tag(nltk_tag):

  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:
    return None


def lemmatize_sentence(sentence, lemmatizer):

  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
  wn_tagged = map(lambda x: (x[0], convert_nltkToWN_tag(x[1])), nltk_tagged)

  res_words = []
  for word, tag in wn_tagged:
    if tag is None:
      res_words.append(word)
    else:
      res_words.append(lemmatizer.lemmatize(word, tag))

  return " ".join(res_words)


def tokenize_sentences(evidences):

    data = []
    for evidence in evidences:
        # iterate through each sentence in the file
        for i in sent_tokenize(evidence):

            temp = []
            # tokenize the sentence into words
            for j in word_tokenize(i):
                temp.append(j.lower())

            data.append(temp)

    return data


##############################################


################## FILTERS ###################


def filter_stopWords(mentions):

    for idx, w in enumerate(mentions):
        if w in set(stopwords.words('english')):
            mentions.remove(mentions[idx])

    return mentions


def filter_singleStrings(mentions):

    lenmentions = []
    for mention in mentions:
        if len(str(mention)) > 2:
            lenmentions.append(mention)

    return lenmentions


def filter_doubles(mentions):

    true_mentions = []
    for idx, mention in enumerate(mentions):
        double = False
        #print(true_mentions)
        #if idx in range(70,90):
        #    print(mention)
        #    print(true_mentions)
        #    print("\n")
        #print(mention)
        if mention in true_mentions:
            #print(str(mention) + " / " + str(tm))
            #double = True
            continue
            #break
        #if double == False:
        else:
            m_arr = str(mention).split(' ')
            if len(m_arr) < 6:
                true_mentions.append(mention)

    return true_mentions


def filter_singleChars(array):

    filter_shorts = []
    for men in array:
        mention = []
        for element in men:
            if len(str(element)) > 2:
                mention.append(element)
        if len(mention) > 0:
            filter_shorts.append(mention)

    return filter_shorts


def filter_dates(mentions):
    for idx, m in enumerate(mentions):
        date = re.search('[0-9]*[0-9][0-9]', str(m))
        if date is not None:
            #print("remove " + bi[idx])
            mentions.remove(mentions[idx])
    return mentions


def filter_stringRubble(bigrams_):

    #clean (can be done better)
    for bi in bigrams_:
        for idx, b in enumerate(bi):
            date = re.search('[0-9]*[0-9][0-9]', b)
            if date is not None:
                #print("remove " + bi[idx])
                bi.remove(bi[idx])

    for bi in bigrams_:
        for idx, b in enumerate(bi):
            if b.find("al.") != -1:
                bi.remove(bi[idx])
            if b.find(" =") != -1:
                bi.remove(bi[idx])
            if len(b) == 1:
                bi.remove(bi[idx])
            if b.find(" et") != -1:
                bi.remove(bi[idx])
            if b.find("-- ") != -1:
                bi.remove(bi[idx])
            if b.find("( ") != -1 or b.find(" )") != -1:
                bi.remove(bi[idx])

    return bigrams_


def filter_unclearChars(mentions):

    mens = []
    for idx, m in enumerate(mentions):
        if str(m).find("al.") == -1:
            mens.append(mentions[idx])
        if str(m).find(" =") == -1:
            mens.append(mentions[idx])
        if len(str(m)) != 1:
            mens.append(mentions[idx])
        if str(m).find(" et") == -1:
            mens.append(mentions[idx])
        if str(m).find("-- ") == -1:
            mens.append(mentions[idx])
        if str(m) != '':
            mens.append(mentions[idx])
        if str(m).find("( ") == -1 or str(m).find(" )") == -1:
            mens.append(mentions[idx])

    return mens


def filter_brokenText(mentions):

    mens=[]
    for mention in mentions:
        split_mention = mention.split(' ')
        encounts = 0
        for w in split_mention:
            if len(w) > 2:
                encounts += 1
        if encounts == len(split_mention):
            mens.append(mention)

    return mens


def filter_starts(mentions):

    mens=[]
    for mention in mentions:
        if mention.startswith("-") or mention.endswith("-"):
            continue
        if mention.startswith(".") or mention.startswith(","):
            continue
        mens.append(mention)

    return mens


def filter_onlyNums(mentions):

    mens=[]
    for mention in mentions:
        if re.search('[a-zA-Z]', mention) == None:
            continue
        if mention.lower().find("proceedings") != -1:
            continue

        mens.append(mention)

    return mens


def filter_spacyOutliers(nlp_base, mentions):

    mens=[]
    for m in mentions:
        split_m = m.split(' ')
        if len(split_m) == 1:
            f_doc = nlp_base(m)
            if len(f_doc.ents) > 0 and f_doc.ents[0].label_ in ['GPE', 'DATE']:
                continue

        mens.append(m)
    return mens


def W2V_filter_soft(model, pointed_mentions):

    #filter soft
    vectors2 = []
    labels2 = []
    for idx, word in enumerate(model.wv.vocab):
        label = word.split(' ')
        count = 0
        for lab in label:
            for mention in pointed_mentions:
                if len(lab) > 3:
                    ment = [x.lower() for x in mention[0]]
                    if lab in ment:
                        double = False

                        for labe in labels2:
                            if labe == word:
                                double = True
                        if double == False:
                            labels2.append(word)
                            vectors2.append(model.wv.vectors[idx])

    return vectors2, labels2


def W2V_filter_hard(model, pointed_mentions):

    #filter hard
    vectors = []
    labels = []
    for idx, word in enumerate(model.wv.vocab):
        label = word.split(' ')
        count = 0
        for lab in label:
            for mention in pointed_mentions:
                if len(lab) > 3:
                    ment = [x.lower() for x in mention[0]]
                    if lab in ment:
                        double = False
                        count += 1
                    else:
                        count = 0
                    if count == len(label):
                        for labe in labels:
                            if labe == word:
                                double = True
                        if double == False:
                            labels.append(word)
                            vectors.append(model.wv.vectors[idx])
        vectors2 = []
        labels2 = []
        for idx, elem in enumerate(vectors):
            count = 0
            for w in elem:
                if w not in set(stopwords.words('english')):
                    count+=1
            if count == len(elem):
                vectors2.append(elem)
                labels2.append(labels[idx])


    return vectors2, labels2


##############################################

######### MENTION CROSSREF AND RANK ##########


#Work In Progress
def spacyBaseDoubleCheck(array, nlp, nlp_base):

    filtered_ents = []
    for ent in array:
        entity = ""
        for el in ent:
            entity = entity + str(el) + " "
        doc = nlp(entity)
        entities = doc.ents
        doc_base = nlp_base(entity)
        entities_base = doc_base.ents
        for nen in entities:
            for bent in entities_base:
                if hasattr(bent, 'label_') and hasattr(ent, 'label_') and (bent.text == ent.text):
                    if bent.label_ in ['PRODUCT', 'ORG', 'NORP', 'WORK_OF_ART']:
                        filtered_ents.append(ent)

                else:
                    filtered_ents.append(ent)

    return filtered_ents


def addPoints(array_orig, array1, array2, points1, points2):

    target = []
    for idx, element in enumerate(array1):
        points = points1
        for idy, double_dummy in enumerate(array_orig):
            #if idy != idx:
                #print(str(element) + "////" + str(double_dummy))
            if element == double_dummy:
                points += 0.2

        for e in element:
            findings = 0
            for comparative in array2:
                if e in comparative:
                    findings += 1
            if findings == len(element):
                points += points2
            if len(element) == 1 and e.isupper():
                #print(e + " pts: " + str(points))
                points = points * 3.5


        tuple = [element, points]
        target.append(tuple)

    for element in array2:
        points = points2
        for e in element:
            findings = 0
            for comparative in array2:
                if e in comparative:
                    findings += 1
            if findings == len(element):
                points += points1

            if len(element) == 1 and e.isupper():
                #print(e)
                points = points * 3.5

        tuple = [element, points]
        target.append(tuple)

    filtered_doubles = []
    for element in target:
        double = False
        for elem in filtered_doubles:
            if element == elem:
                double = True
                break
        if double == False:
            filtered_doubles.append(element)

    return filtered_doubles


def mentionRankThreshold(pointed_mentions):

    tris = []
    for pm in pointed_mentions:
        if pm[1] > 1.6:
            tris.append(pm[0])
    return tris


##############################################

############### MODEL CONVERTERS #############


def convertSentsToBiTriGs(evidences):

    sentence_stream = [doc.split(" ") for doc in evidences]

    #sentence_stream=brown_raw[0:10]
    bigram = Phrases(sentence_stream, min_count=1, delimiter=b' ')
    trigram = Phrases(bigram[sentence_stream], min_count=1, delimiter=b' ')
    bigrams_ = []
    trigrams_ = []

    for sent in sentence_stream:
        bigrams_.append([b for b in bigram[sent] if b.count(' ') == 1])
        trigrams_.append([t for t in trigram[bigram[sent]] if t.count(' ') == 2])

    return bigrams_, trigrams_


##############################################


def getMentionsFromArgs(sentences, nlp, nlp_base):

    mentions = []
    for idx, sentence in enumerate(sentences):

        text = word_tokenize(sentences[idx])
        tokens = nltk.pos_tag(text)

        tags = [lis[1] for lis in tokens]
        texts = [lis[0] for lis in tokens]


        doc = nlp(sentence)
        entities = doc.ents
        doc_base = nlp_base(sentence)
        entities_base = doc_base.ents

        precandidates = []
        candidates = []
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

        #print(entities_final)
        #for idx, tag in enumerate(tags):
        #    if tag == 'NN' or tag.startswith('JJ'):
        #        #print(idx)
        #        precandidates.append(texts[idx])
        #        if len(texts) > idx+1:
        #            candidates.append(texts[idx+1])
        #print(candidates)
        #for candidate in precandidates:
        #    for ent in entities_final:
        #        if hasattr(ent, 'label_'):
        #            if ent.text.find(candidate):
        #                mentions.append(ent)
        #print(entities_final)
        ents = filter_doubles(entities_final)
        if len(ents) > 0:
            for e in ents:
                mentions.append(e.text)
    #print(mentions)
        #mentions.append()

    return mentions


def mentionsfromSents(sentences, nlp, nlp_base):

    mentions = []
    for idx, sentence in enumerate(sentences):

        text = word_tokenize(sentences[idx])
        tokens = nltk.pos_tag(text)

        tags = [lis[1] for lis in tokens]
        texts = [lis[0] for lis in tokens]


        doc = nlp(sentence)
        entities = doc.ents
        doc_base = nlp_base(sentence)
        entities_base = doc_base.ents

        precandidates = []
        candidates = []
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
                if len(texts) > idx+1:
                    candidates.append(texts[idx+1])

        for candidate in candidates:
            for ent in entities:
                if hasattr(ent, 'label_'):
                    if ent.text.find(candidate):
                        mentions.append(ent)

    return mentions


def getSemanticMentions(evidences):

    semantic_mentions = []
    for idt, evidence in enumerate(evidences):

        text = word_tokenize(evidences[idt])
        tokens = nltk.pos_tag(text)

        tags = [lis[1] for lis in tokens]
        texts = [lis[0] for lis in tokens]

        #phrases = []
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
            if tags[idx] == 'PRP':
                subject = True
                index.append(idx)

            elif tags[idx] == 'DT' and tags[idx+1].startswith('NN'):
                subject = True
                index.append(idx)
                index.append(idx+1)

            elif tags[idx].startswith('NN'):
                subject = True
                index.append(idx)

            end = len(index)
            if subject:
                for idy, t in enumerate(tags):
                    if idy > index[end-1]:
                        if t.startswith('VB'):
                            verbal = True
                            index2.append(idy)
                            id = idy
                            tagloop = tags[id+1:]
                            for idz, tl in enumerate(tagloop):
                                while not other:
                                    if tagloop[idz].startswith('VB'):
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
                                index3.append(idz)
                                break
                            elif len(tags) > (idz + 1):
                                if tt == 'DT' and ( tags[idz+1].startswith('NN') or tags[idz+1].startswith('JJ')):
                                    methodstart = True
                                    index3.append(idz)
                                    index3.append(idz+1)
                                    break
                                elif tt not in ['PRP$', 'IN']:
                                    methodstart = False
                                    break

                    if methodstart:
                        id = index3[len(index3) - 1]
                        methodstart_id = index3[len(index3) - 1]
                        tagloop = tags[id+1:]
                        for idw, tl in enumerate(tagloop):
                            while not other:
                                if tagloop[idw].startswith('NN'):
                                    proper_method = True

                                if tagloop[idw].startswith('NN') or tagloop[idw].startswith('JJ'):
                                    index3.append(id + (idw+1))
                                    if (id+idw+1) == (len(texts)-1):
                                        other = True
                                        break
                                    else:
                                        idw += 1
                                else:
                                    other = True
                                    break

            if proper_method:
                mmention = []
                for ind in index3:
                    if(ind >= methodstart_id):
                        mmention.append(texts[ind])
                if len(mmention) < 6:
                    semantic_mentions.append(mmention)

                continuation = index3[(len(index3) - 1)] + 1

    return semantic_mentions


def getRankedMentions(lemmatized, nlp, nlp_base):


    mentions_orig = getMentionsFromArgs(lemmatized, nlp, nlp_base)
    mentions = filter_doubles(mentions_orig)
    #print(pointed_mentions)

    mentions = filter_singleStrings(mentions)
    mentions = filter_brokenText(mentions)
    mentions = filter_starts(mentions)
    mentions = filter_onlyNums(mentions)
    #mentions = filter_unclearChars(mentions)
    mentions = filter_stopWords(mentions)
    true_mentions = filter_spacyOutliers(nlp_base, mentions)


    #print(true_mentions)
    #true_mentions = filter_doubles(mentions)
    #print(pointed_mentions)

    semantic_mentions = getSemanticMentions(lemmatized)

    #true_mentions = filter_singleChars(mentions)
    semantic_mentions = filter_singleChars(semantic_mentions)

    pointed_mentions = addPoints(mentions_orig, true_mentions, semantic_mentions, 1.4, 1.6)

    pointed_mentions = sorted(pointed_mentions, key=lambda tup: tup[1], reverse=True)


    f = open('./outs/pointed_mentions.txt', "w")
    f.write("List of scored Method/Technologies")
    f.write("\n")
    f.write("-----------------------------------")
    f.write("\n")
    f.write("\n")

    for p in pointed_mentions:
        f.write(str(p))
        f.write("\n")

    f.close()

    return pointed_mentions
