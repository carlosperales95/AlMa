# coding: utf-8

import sys
import string
from os import listdir
from os.path import isfile, join
from utils import *
from parsingUtils import *
from xmlUtils import *


if len(sys.argv) == 1:
    print("\n")
    print("Input file or directory not specified.")
    print("Need at least one argument to be passed.")
    print("Warning -- If output file or directory is not a specified parameter, the filename 'out_file.txt' will be used. ")
    print("Make sure there is no file with the same name as it would be overwritten.")
    print("\n")
    print("Warning -- For running the script with all the test corpus, add the parameter -test")
    print("Try again.")
    sys.exit()



if len(sys.argv) < 3:
    print("\n")
    print("(Warning) Output file parameter not specified, filename 'out_file.txt' will be used. ")
    print("Any existing file in the directory with that filename will be overwritten.")
    print("\n")
    output_name = "out_file.txt"
else:
    output_name = str(sys.argv[2])



if str(sys.argv[1]) == "-test":
    onlyfiles = [f for f in listdir("./test_corpus/") if isfile(join("./test_corpus/", f))]
    dir="./test_corpus/"
else:
    if str(sys.argv[1]) == "-batch":
        onlyfiles = [f for f in listdir("./full_batch/") if isfile(join("./full_batch/", f))]
        dir="./full_batch/"
    else:
        onlyfiles = []
        onlyfiles.append(str(sys.argv[1]))
        dir=""

filen = open("INTROS.txt", "w")

for idx,file in enumerate(onlyfiles):


    if len(onlyfiles) > 0:
        output_name = "./sum_batch/sum_"+file[:-4]+".txt"

    if file.find("word.docx.txt") != -1:
        xml_name = "./batch_xml/" + file[:file.find("word.docx.txt")] + ".xml"
    else:
        xml_name = "./batch_xml/" + file[:-4] + ".xml"


    sections = xmlToSections(xml_name)

    filen.write(file)
    filen.write("\n")

    for section in sections:
        filen.write(section)
        filen.write("\n")

    filen.write("\n")
    filen.write("--------------------------------------")
    filen.write("\n")
    filen.write("\n")


    f = open(dir+file, "r")
    f2 = open(output_name, "w")

    paper_full =f.read()

    printable = set(string.printable)


    print("Starting paper " + file[:-4])
    print("Found Abstract .... (1/4)")

    paper_abstract = getAbstract(paper_full)

    paper_abstract = filter(lambda x: x in printable, paper_abstract)


    if paper_abstract[len(paper_abstract)-1:] != ".":
            paper_abstract = paper_abstract + "."

    paper_abstract = cleanTextRubble(paper_abstract)
    paper_abstract =  paper_abstract.replace('   ', ' ').replace('  ', ' ')

    f2.write(paper_abstract)
    f2.write("\n")
    f2.write("\n")

    print("Writing .... (2/4)")

    intro = getSection(paper_full, sections[0], sections[1])
    if intro.endswith(' 2 ') or intro.endswith(' 2.') or intro.endswith(' 2. ') or intro.endswith(' 2.\t') or intro.endswith(' 2  ') or intro.endswith(' 2\t'):
        intro = intro[:-3]


    intro = filter(lambda x: x in printable, intro)

    if intro[len(intro)-1:] != ".":
            intro = intro + "."

    intro = cleanTextRubble(intro)
    intro =  intro.replace('   ', ' ').replace('  ', ' ')

    f2.write(intro)
    f2.write("\n")
    f2.write("\n")

    #intro = getIntro(paper_full)


    print("Found Conclusion/Discussion .... (3/5)")

    temp_conclusion = getConclusion(paper_full)

    print("Cleaning Conclusion/Discussion .... (4/5)")

    conclusion = figureHunter(temp_conclusion)
    conclusion = removeFigureTags(conclusion, "Figure ")
    conclusion = removeFigureTags(conclusion, "Table ")
    #conclusion = conclusion[:conclusion.rfind('\n')]

    #f2.write(conclusion.replace('\n', ' ').replace('\r', ' ').replace('   ', ' ').replace('- ', '').replace('  ', ' '))
    conclusion = conclusion.replace('- \r', '').replace('-\r', '').replace(' \n', ' ').replace(' \r', ' ').replace('\n', '').replace('\r', '').replace('   ', ' ').replace('- ', '').replace('  ', ' ').replace('\t', '')

    conclusion = filter(lambda x: x in printable, conclusion)

    if conclusion[len(conclusion)-1:] != ".":
            conclusion = conclusion + "."

    conclusion = cleanTextRubble(conclusion)
    conclusion =  conclusion.replace('   ', ' ').replace('  ', ' ')

    f2.write(conclusion)
    f2.write("\n")

    print("Writing .... (5/5)")

    f.close()
    f2.close()

    print("Paper " + file[:-4] + " finished.")
    print("\n")


print("Script finished, paper(s) summarized")
filen.close()

#text = """
# * Corresponding [5][6687] author. translation (SMT) systems.       .English and Chinese graphs, simply 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0 Tail Head Ours  Google.  1 http://people.entitycube. com 430 Proceedings of the 2010 Conference [c] on Empirical Methods in Natural Language [57] Processing. Section 2 reviews exi                          sting work. Section 3 then develops our framework. Section 4 reports (667) experimental results (5) and Section 5 concludes our work. (a) English PeopleEntityCube Ge. Such engine ∗This work was done when the first two authors visited Mi-    crosoft Research Asia.
#https://stackoverflow.com/questions/6883049/regex-to-extract-urls-from-href-attribute-in-html-with-python
#
 #On NIST08 Chinese-English translation task, we obtain an improvement of 0.48 BLEU from a competitive baseline (30.01 BLEU to 30.49 BLEU) with the Stanford Phrasal MT system. 1393 Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1393–1398, Seattle, Washington, USA, 18-21 October 2013. Qc 2013 Association for Computational Linguistics
 #"""

#print(text)
#print("\n")
#text = cleanTextRubble(text)
#print(text)
