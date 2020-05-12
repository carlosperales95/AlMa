import pke

# initialize keyphrase extraction model, here TopicRank
extractor = pke.unsupervised.TopicRank()

# load the content of the document, here document is expected to be in raw
# format (i.e. a simple text file) and preprocessing is carried out using spacy
extractor.load_document(input='./full_batch/C10-1070word.docx.txt', language='en')

# keyphrase candidate selection, in the case of TopicRank: sequences of nouns
# and adjectives (i.e. `(Noun|Adj)*`)
extractor.candidate_selection()

# candidate weighting, in the case of TopicRank: using a random walk algorithm
extractor.candidate_weighting()

# N-best selection, keyphrases contains the 10 highest scored candidates as
# (keyphrase, score) tuples
keyphrases = extractor.get_n_best(n=10)

print(keyphrases)
print("\n")



import pke

# define the set of valid Part-of-Speeches
pos = {'NOUN', 'PROPN', 'ADJ'}

# 1. create a TextRank extractor.
extractor = pke.unsupervised.TextRank()

# 2. load the content of the document.
extractor.load_document(input='./full_batch/C10-1070word.docx.txt',
                        language='en',
                        normalization=None)

# 3. build the graph representation of the document and rank the words.
#    Keyphrase candidates are composed from the 33-percent
#    highest-ranked words.
extractor.candidate_weighting(window=2,
                              pos=pos,
                              top_percent=0.33)

# 4. get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)

print(keyphrases)
print("\n")


import pke

# define the set of valid Part-of-Speeches
pos = {'NOUN', 'PROPN', 'ADJ'}

# 1. create a SingleRank extractor.
extractor = pke.unsupervised.SingleRank()

# 2. load the content of the document.
extractor.load_document(input='./full_batch/C10-1070word.docx.txt',
                        language='en',
                        normalization=None)

# 3. select the longest sequences of nouns and adjectives as candidates.
extractor.candidate_selection(pos=pos)

# 4. weight the candidates using the sum of their word's scores that are
#    computed using random walk. In the graph, nodes are words of
#    certain part-of-speech (nouns and adjectives) that are connected if
#    they occur in a window of 10 words.
extractor.candidate_weighting(window=10,
                              pos=pos)

# 5. get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)

print(keyphrases)
print("\n")


import pke

# define the valid Part-of-Speeches to occur in the graph
pos = {'NOUN', 'PROPN', 'ADJ'}

# define the grammar for selecting the keyphrase candidates
grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

# 1. create a PositionRank extractor.
extractor = pke.unsupervised.PositionRank()

# 2. load the content of the document.
extractor.load_document(input='./full_batch/C10-1070word.docx.txt',
                        language='en',
                        normalization=None)

# 3. select the noun phrases up to 3 words as keyphrase candidates.
extractor.candidate_selection(grammar=grammar,
                              maximum_word_number=3)

# 4. weight the candidates using the sum of their word's scores that are
#    computed using random walk biaised with the position of the words
#    in the document. In the graph, nodes are words (nouns and
#    adjectives only) that are connected if they occur in a window of
#    10 words.
extractor.candidate_weighting(window=10,
                              pos=pos)

# 5. get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)

print(keyphrases)
print("\n")
