text = sample_file.read()


nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

for ent in doc.ents:
  print(ent.label_, ' | ', ent.text)


print("\n")
print("\n")
print("\n")




nlp = spacy.load('mymodel')
doc = nlp(text)

for ent in doc.ents:
  print(ent.label_, ' | ', ent.text)


print("\n")
print("\n")
print("\n")


            for idy, ent in enumerate(doc.ents):
                    if ent[idy].text.find(texts[idx+1]):
                        if hasattr(ent[idy], "label_"):
                            f.write(evidence)
                            f.write("\n")
                            f.write(str(tokens))
                            f.write("\n")
                            f.write(ent[idy].label_ + "|" + ent[idy].text)
                            f.write("\n")
                            f.write("\n")
                            f.write("\n")


    sentence = paper_abstract.lower()

    # onegrams = OneGramDist(filename='count_10M_gb.txt')
    onegrams = OneGramDist(filename='count_1M_gb.txt.gz')
    # onegrams = OneGramDist(filename='count_1w.txt')
    onegram_fitness = functools.partial(onegram_log, onegrams)
    paper_abstract = segment(sentence, word_seq_fitness=onegram_fitness)
