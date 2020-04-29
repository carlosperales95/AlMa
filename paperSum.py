import sys
from os import listdir
from os.path import isfile, join
from utils import *
from parsingUtils import *


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


for idx,file in enumerate(onlyfiles):

    if len(onlyfiles) > 0:
        output_name = "./sum_batch/sum_"+file[:-4]+".txt"

    f = open(dir+file, "r")
    f2 = open(output_name, "w")

    paper_full =f.read()

    print("Starting paper " + file[:-4])
    print("Found Abstract .... (1/4)")

    paper_abstract = getAbstract(paper_full)

    f2.write(paper_abstract)
    f2.write("\n")
    f2.write("\n")

    print("Writing .... (2/4)")

    intro = getIntro(paper_full)

    f2.write(intro)
    f2.write("\n")
    f2.write("\n")

    find_index = paper_full.find('Conclusion')

    if find_index == -1:
        find_index = paper_full.find('Discussion')

    paper_temp = paper_full[find_index:]
    find_index_2 = find_index + paper_temp.find('\n')

    line_conclusion = paper_full[find_index:find_index_2] + "\n"

    print("Found Conclusion/Discussion .... (3/5)")

    temp_conclusion = getConclusion(paper_full)

    print("Cleaning Conclusion/Discussion .... (4/5)")

    conclusion = figureHunter(temp_conclusion)
    conclusion = conclusion[:conclusion.rfind('\n')]

    f2.write(line_conclusion + conclusion.replace('\n', ' ').replace('\r', ' ').replace('   ', ' ').replace('- ', '').replace('  ', ' '))
    f2.write("\n")

    print("Writing .... (5/5)")

    f.close()
    f2.close()

    print("Paper " + file[:-4] + " finished.")
    print("\n")


print("Script finished, paper(s) summarized")
