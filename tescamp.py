import sys
import os
from os import listdir
from os.path import isfile, join


init_block = ""

block = 0
paper_title = "el titulo\nvamo\na\n\n\nver\nxaba\nquenoteente\nra\nsae\n"
paper_cut = paper_title.find("\n")

paper_doubt = paper_title[paper_cut:]
while block < 3:

    paper_doubt_temp = paper_doubt.find("\n")
    print(paper_doubt)
    print("CIT: ")
    print(paper_doubt_temp)

    if paper_doubt_temp > 1:
        block += 1

    else:
        block = 0
        init_block = paper_doubt[paper_doubt_temp:]

    print("BL: ")
    print(block)
    paper_doubt2 = paper_doubt[paper_doubt_temp+1:]
    paper_doubt = paper_doubt2
    print("l'bloq: ")
    print(init_block)


print("FISISH")
index = paper_title.find(init_block)
print("\n" + paper_title[:index])
