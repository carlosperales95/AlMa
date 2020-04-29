import sys
import os
from os import listdir
from os.path import isfile, join
import shutil
from utils import *
from parsingUtils import *





if len(sys.argv) < 2:
    print("\n")
    print("Topic query or not specified.")
    print("Need at least 1 argument to be passed.")
    print("Try python searchByTopyc.py <Topic query> ")
#    print("Warning -- For running the script with all the test corpus, add the parameter -test")
    print("Try again.")
    sys.exit()


topic = str(sys.argv[1]).lower()
if topic.find("_") != -1:
    topic = topic.replace("_", " ")

corpus_path = "../scisumm-corpus/data/"
txt_folder_name = "Documents_TXT"


print("Deleting previous batch......")
print("\n")
cleanBatch("./full_batch/")
cleanBatch("./sum_batch/")


txt_files = scisummFindTXT(corpus_path, txt_folder_name)

###SEARCH TOPIC IN TITLE
topic_papers = []
f2 = open("titles.txt", "w")



for file in txt_files:

    f = open(file, "r")
    paper_full =f.read()
    paper_title = findTitle(paper_full)

    if paper_title.lower().find(topic) != -1:

        topic_papers.append(file)
        f2.write("- " + paper_title.replace("\n", " ") + "\n")
        f2.write("---------------------------------------------------------------------------------\n")
        #print(paper_title + "\nOTHER PAPER\n\n\n")
    f.close()
f2.close()


print("Papers based on "+topic+": "+str(len(topic_papers))+"\n\n")

if(len(topic_papers) > 0):
    for paper in topic_papers:
        print(paper)
        filename = paper[ (paper.find("_TXT/") + 5) : ]
        newPath = shutil.copy(paper, './full_batch/' + filename)
        print("Moving paper to full_batch folder...")
        print("\n")
    #if index_title != -1:
        #if index_title >= 90:
        #    paper_title = paper_temp[:index_title]
            #print("**"+paper_title+"\n\n\n\n")
        #else :
        #    if index_title < 50:
        #        paper_temp2 = paper_temp[index_title:]
        #        index_title2 = paper_temp2.find("\n\n")
        #        paper_title = paper_temp[:index_title2]
                #print(paper_title)
    #dirs = walkfs(corpus_dir, "Documents_TXT/")
    #print(dirs[2])
