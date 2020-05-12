import spacy
import random


TRAIN_DATA = [('Search Analytics: Business Value & BigData NoSQL Backend, Otis Gospodnetic ', {'entities': [(35,42,'TECH'), (43,48,'TECH')]}),
              ('Introduction to Elasticsearch by Radu', {'entities': [ (16,29,'TECH')]}),
              ('Our approach joins MySQL and SVM', {'entities': [(19, 24, 'TECH'), (29, 32, 'TECH')]}),
              ('Terasort been proved to work much more efficiently than its predecessors  ', {'entities': [(0, 8, 'TECH')]}),
              ('In this paper we compare Spark and Hadoop', {'entities': [(25, 30, 'TECH'), (35, 41, 'TECH')]}),
              ('That is why we believe SVM can work much better in this scenario', {'entities': [(23, 26, 'TECH')]}),
              ('We used SpaCy for training the model and recognizing entities', {'entities': [(8, 13, 'TECH')]}),
              ('Unfortunately, there were lots of compatibility issues with Python', {'entities': [(60, 66, 'TECH')]}),
              ('DeepWalk: Online Learning of Social Representations; Perozzi et al. 2014', {'entities': [(0, 8, 'TECH'), (10, 25, 'METHOD')]}),
              ('In practice, DeepWalk will perform multiple random walks on each node to generate a large corpus', {'entities': [(13, 21, 'TECH'), (35, 56, 'METHOD')]}),
              ('SkipGram is an algorithm that is used to create word embeddings', {'entities': [(0, 8, 'TECH'), (48, 63, 'METHOD')]}),
              ('In the process of topic segmentation, a semi-supervised text-clustering model, the Labeled Dirichlet Multi Mixture, is used to integrate domain knowledge into technological segmentation processes', {'entities': [(18, 36, 'METHOD'), (40, 77, 'METHOD'), (83, 114, 'TECH')]}),
              ('For Python users, there is an easy-to-use keyword extraction library called RAKE, which stands for Rapid Automatic Keyword Extraction', {'entities': [(4, 10, 'TECH'), (42, 60, 'METHOD'), (76, 80, 'TECH'), (99, 134, 'TECH')]}),
              ('When importing a file, Python only searches the current directory', {'entities': [(23, 29, 'TECH')]}),
              ('TextRank is an algorithm based on PageRank, which often used in keyword extraction and text summarization', {'entities': [(0, 8, 'TECH'), (34, 42, 'TECH'), (64, 82, 'METHOD'), (87, 105, 'METHOD')]}),
              ('Multiplying these two quantities provides the TF-IDF score of a word in a document' , {'entities': [(46, 72, 'TECH')]}),
              ('TD-IDF algorithms have several applications in machine learning', {'entities': [(0, 6, 'TECH'), (47, 63, 'METHOD')]}),
              ('CRF model is a new probabilistic model for segmenting and labeling sequence data', {'entities': [(0, 3, 'TECH'), (19, 38, 'METHOD'), (58, 80, 'METHOD')]}),
              ('In Chapter 3, we mainly presented the BioASQ project and its dataset', {'entities': [(38, 44, 'TECH')]}),
              ('To extract such patterns, we first used the Genia tagger', {'entities': [(44, 56, 'TECH')]}),
              ('Automatic keyword extraction (AKE) is the task to identify a small set of words, key phrases, keywords, or key segments from a document that can describe the meaning of the document', {'entities': [(0, 28, 'METHOD'), (30, 33, 'METHOD'), (58, 80, 'METHOD')]}),
              ('Cohen uses N-Gram statistical information to automatic index the document', {'entities': [(11, 17, 'METHOD')]}),
              ('This approach includes Naïve Bayes, SVM, Bagging, etc.', {'entities': [(23, 34, 'TECH'), (36, 39, 'TECH'), (41, 48, 'METHOD')]}),
              ('Inverse Document Frequency is a measure of general importance of term obtained by dividing number of all documents by number of documents', {'entities': [(0, 26, 'METHOD')]}),
              ('Next is to stem the words using Krovetz Algorithm based on WordNet Dictionary', {'entities': [(32, 49, 'METHOD'), (59, 66, 'TECH')]}),
              ('Using TF-IDF variants, there are six different values for every word and filtering can be done by using cross-domain comparison', {'entities': [(6, 13, 'TECH'), (104, 127, 'METHOD')]}),
              ('We perform clustering algorithms such as K-means in the topic space to obtain clusters', {'entities': [(11, 32, 'METHOD'), (41, 48, 'TECH')]}),
              ('This method uses a CRFs model to label keyphrases in text', {'entities': [(19, 23, 'TECH'), (33, 49, 'METHOD')]})]


def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)


    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


prdnlp = train_spacy(TRAIN_DATA, 32)

# Save our trained Model
modelfile = input("Enter your Model Name: ")
prdnlp.to_disk(modelfile)

#Test your text
test_text = input("Enter your testing text: ")
doc = prdnlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
