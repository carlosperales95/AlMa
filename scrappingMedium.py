import spacy
from spacy import displacy

# install spacy model
#! pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.2.0/en_core_web_md-2.2.0.tar.gz
nlp = spacy.load("en_core_web_md")
doc = nlp("I think Barack Obama met founder of Facebook at occasion of a release of a new NLP algorithm.")

displacy.render(doc, style="dep") # (1)
displacy.render(doc, style="ent") # (2)

# import custom modules
import html_file_parsing as hfp

# define listing sources
Tech_Topics=[
    'https://medium.com/topic/artificial-intelligence',
    'https://medium.com/topic/blockchain',
    'https://medium.com/topic/cryptocurrency',
    'https://medium.com/topic/data-science',
    'https://medium.com/topic/machine-learning',
    'https://medium.com/topic/neuroscience',
    'https://medium.com/topic/programming',
    'https://medium.com/topic/self-driving-cars',
    'https://medium.com/topic/software-engineering',
    'https://medium.com/topic/technology' ]

# call the class
at = hfp.ArticleTable(Tech_Topics)
multiple_articles_table_df = at.main() # Compute a Pandas dataframe to write into multiple_articles_table


TRAIN_DATA_ALL =list(train_table.apply(lambda x : mark_targets(x, ['ORG', 'PERSON', 'LOC', 'TOPIC', 'GPE','DATE', 'EVENT', 'WORK_OF_ART'], "sents", ['ORG', 'PERSON', 'LOC', 'TOPIC', 'GPE','DATE', 'EVENT', 'WORK_OF_ART']), axis=1))


from evalmodel import test_eval_model

dico, df_metrics = test_eval_model("ner_models/mytrainedmodel", TEST_DATA)
