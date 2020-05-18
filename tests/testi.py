import spacy


text = "In their seminal paper on SMT , Brown and his colleagues highlighted the problems we face as we go from IBM Models 1-2 to 3-5 ( Brown et al. , 1993 ) 3 : `` As we progress from Model 1 to Model 5 , evaluating the expectations that gives us counts becomes increasingly difficult ."

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

print(doc)

for ent in doc.ents:
  print(ent.label_, ' | ', ent.text)
