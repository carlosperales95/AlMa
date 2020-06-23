import sys
import os
from os import listdir
from os.path import isfile, join
import shutil
from utils import *
from parsingUtils import *
from xmlUtils import *




if len(sys.argv) < 2:
    print("\n")
    print("Topic query or not specified.")
    print("Need at least 1 argument to be passed.")
    print("Try python searchByTopyc.py <Topic query> ")
#    print("Warning -- For running the script with all the test corpus, add the parameter -test")
    print("Try again.")
    sys.exit()


topics = []

for idx, arg in enumerate(sys.argv):

    if idx > 0:
        topics.append(str(sys.argv[idx]).lower())
    else:
        continue


corpus_path = "../scisumm-corpus/data/"
txt_folder_name = "Documents_TXT"
xml_folder_name = "Reference_XML"


print("Deleting previous batch......")
print("\n")
cleanBatch("./full_batch/")
cleanBatch("./sum_batch/")


txt_files = scisummFindTXT(corpus_path, txt_folder_name)
xml_files = scisummFindXML(corpus_path, xml_folder_name)

###SEARCH TOPIC IN TITLE
topic_papers = []
topic_xml = []
f2 = open("titles.txt", "w")


for file in txt_files:

    f = open(file, "r")
    paper_full =f.read()
    paper_title = findTitle(paper_full)

    for idx, topic in enumerate(topics):

        if paper_title.lower().find(topics[idx]) != -1:

            filename_start_idx = file.find("Documents_TXT/")
            filename_end_idx = file.find("word.docx.txt")
            if filename_end_idx == -1:
                filename_end_idx = file.find(".docx.txt")
                if filename_end_idx == -1:
                    filename_end_idx = file.find(".txt")

            filename = file[filename_start_idx + len("Documents_TXT/") : filename_end_idx]
            for xmlfile in xml_files:
                if xmlfile.find(filename) != -1:
                    topic_xml.append(xmlfile)

            topic_papers.append(file)
            f2.write("- " + paper_title.replace("\n", " ") + "  #"+file+ "\n")
            f2.write("---------------------------------------------------------------------------------\n")
            #print(paper_title + "\nOTHER PAPER\n\n\n")
            break
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

if(len(topic_xml) > 0):
    for xmlpaper in topic_xml:
        filename = xmlpaper[ (xmlpaper.find("_XML/") + 5) : ]
        newPath = shutil.copy(xmlpaper, './batch_xml/' + filename)
        print("Moving xml to full_batch folder...")
        print("\n")






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
