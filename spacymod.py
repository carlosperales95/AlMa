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
              ('This method uses a CRFs model to label keyphrases in text', {'entities': [(19, 23, 'TECH'), (33, 49, 'METHOD')]}),
              ('For example, the current-best error rate on the MNIST digit-recognition task (<0.3%) approaches human performance', {'entities': [(48, 53, 'TECH'), (54, 71, 'METHOD')]}),
              ('We  are  not  the  first  to  consider  alternatives  to  traditional neuron models in CNNs.', {'entities': [(70, 83, 'METHOD'), (87, 108, 'TECH')]}),
              ('Max-pooling layers, of the kind described in Section 3.4, follow both response normalization layers as well as the fifth convolutional layer', {'entities': [(0, 18, 'METHOD'), (70, 99, 'METHOD'), (121, 140, 'METHOD')]}),
              ('The ReLU non-linearity is applied to the output of every convolutional and fully connected layer', {'entities': [(4, 8, 'TECH'), (9, 22, 'METHOD'), (81, 96, 'METHOD')]}),
              ('Many current NLP systems and techniques treat words as atomic unit', {'entities': [(13, 16, 'TECH')]}),
              ('Another interesting architecture of NNLM was presented', {'entities': [(36, 40, 'TECH')]}),
              ('Then the word vectors are first learned using neural network with a single hidden layer', {'entities': [(9, 21, 'METHOD'), (46, 60, 'TECH'), (75, 87, 'METHOD')]}),
              ('LDA moreover becomes computationally very expensive on large data sets', {'entities': [(0, 3, 'TECH')]}),
              ('Recurrent neural network based language model has been proposed to overcome certain limitations of the feedforward NNLM, ', {'entities': [(0, 24, 'TECH'), (31, 45, 'METHOD'), (103, 119, 'TECH')]}),
              ('We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest', {'entities': [(20, 53, 'TECH'), (112, 120, 'TECH')]}),
              ('ImageNet is a dataset of over 15 million labeled high-resolution images belonging to roughly 22,000 categories', {'entities': [(0, 8, 'TECH')]}),
              ('We define Markov chains over various image maps, and treat the equilibrium distribution over map locations as activation and saliency values', {'entities': [(10, 23, 'TECH'), (63, 87, 'METHOD'), (125, 140, 'METHOD')]}),
              ('Given access to the same feature information, GBVS predicts human fixations more reliably than the standard algorithms', {'entities': [(46, 50, 'TECH')]}),
              ('This paper proposes a Fast Region-based Convolutional Network method (Fast R-CNN) for object detection', {'entities': [(22, 61, 'TECH'), (70, 80, 'TECH'), (86, 102, 'METHOD')]}),
              ('For SVM and bounding-box regressor training, features are extracted from each object proposal in each image and written to disk', {'entities': [(4, 7, 'TECH'), (12, 43, 'TECH')]}),
              ('With very deep networks, such as VGG16, this process takes 2.5 GPU-days for the 5k images of the VOC07 trainval set', {'entities': [(33, 38, 'TECH')]}),
              ('We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic  objective  functions,  based  on  adaptive  estimates  of  lower-order  moments', {'entities': [(13, 17, 'TECH'), (48, 75, 'METHOD'), (79, 111, 'METHOD'), (125, 144, 'METHOD')]}),
              ('Our method is designed to combine the advantages of two recently popular methods: AdaGrad (Duchi et al., 2011), which works well with sparse gradients, and RMSProp (Tieleman & Hinton, 2012), which works well in on-line and non-stationary settings', {'entities': [(82, 89, 'TECH'), (134, 150, 'METHOD'), (156, 163, 'TECH'), (223, 246, 'METHOD')]}),
              ('Plugging the appropriate matrix from the above equation into var1 or var2 gives rise to our ADAGRAD family of algorithms', {'entities': [(92, 99, 'TECH'), (100, 120, 'METHOD')]}),
              ('Informally, FTRL methods choose the best decision in hindsight at every iteration', {'entities': [(12, 16, 'TECH')]})
]



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
