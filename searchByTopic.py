import sys
import os
from os import listdir
from os.path import isfile, join



if len(sys.argv) < 1:
    print("\n")
    print("Topic query or not specified.")
    print("Need at least 1 argument to be passed.")
    print("Try python searchByTopyc.py <Topic query> ")
#    print("Warning -- For running the script with all the test corpus, add the parameter -test")
    print("Try again.")
    sys.exit()


#query = str(sys.argv[1])
corpus_path = "../scisumm-corpus/data/"
topic = "Translation"

txt_dirs = []
# r=root, d=directories, f = files
for r, d, f in os.walk(corpus_path):
    for dir in d:
        if str(dir).find("Documents_TXT") == 0:
            txt_dirs.append(os.path.join(r, dir))


txt_files = []
for dir in txt_dirs:
    count = 0
    temp = []
    for r, d, f in os.walk(dir):
        for file in f:
            if str(file).find("word") == -1:
                count = count + 1
                temp.append(str(dir)+"/"+str(file))

            else:
                txt_files.append(str(dir)+"/"+str(file))

        if len(f) == count:
            for tf in temp:
                txt_files.append(tf)


###SEARCH TOPIC IN TITLE
topic_papers = []
f2 = open("titles.txt", "w")

for file in txt_files:
    f = open(file, "r")
    paper_full =f.read()

    #paper_temp = paper_full[:200]
    index_title = paper_full.find("Abstract")
    if index_title == -1:
        index_title = paper_full.find("ABSTRACT")

    if index_title == -1:
        index_title = paper_full.find("Introduction")

    if index_title == -1:
        index_title = paper_full.find("INTRODUCTION")

    paper_temp = paper_full[index_title:]
    find_index_2 = index_title + paper_temp.find('\n')

    paper_title = paper_full[:find_index_2]

    paper_cut_start = paper_title.find("\n")
    paper_title2 = ""
    while paper_cut_start < 2:
            paper_cut_start = 1
            paper_title2 = paper_title[paper_cut_start:]
            paper_title = paper_title2
            paper_cut_start = paper_title.find('\n')
            #print(paper_cut_start)

            if paper_cut_start == -1:
                break


    paper_title = paper_title[0:]
    paper_cut = paper_title.find("\n")

    paper_doubt = paper_title[paper_cut:]
    paper_doubt_cut = 0

    block = 0
    init_block = "empty"
    #print(paper_doubt)

    while block < 2:

        paper_doubt_temp = paper_doubt.find("\n")
        #print(paper_doubt_temp)
        #print(paper_doubt)

        #print(paper_doubt)
        #print("CIT: ")
        #print(paper_doubt_temp)

        if paper_doubt_temp > 1:
            block += 1

        else:
            block = 0
            init_block = paper_doubt[paper_doubt_temp:]

        if paper_doubt_temp == -1:
            break

        #print("BL: ")
        #print(block)
        paper_doubt2 = paper_doubt[paper_doubt_temp+1:]
        paper_doubt = paper_doubt2
        #print("l'bloq: ")
        #print(init_block)

    index = paper_title.find(init_block)
    paper_title =  paper_title[:index]

    res = [i for i in range(len(paper_title)) if paper_title.startswith("\n", i)]
    #print("The start indices of the substrings are : " + str(res))

    gap = 0
    paper_title2 = ""
    for idx,r in enumerate(res):
        if idx < len(res)-1:
            if res[idx+1]-res[idx] == 2:
                gap += 1
            else:
                gap = 0

            if gap == 2:
                paper_title2 = paper_title[:res[idx]]
                break

    paper_title = paper_title2
    if paper_title.find(topic) != -1:
        topic_papers.append(file)
        f2.write("- " + paper_title + "\n")
        f2.write("---------------------------------------------------------------------------------\n")
        #print(paper_title + "\nOTHER PAPER\n\n\n")
    f.close()
f2.close()


print("Papers based on "+topic+": "+str(len(topic_papers)))





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
