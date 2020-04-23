import sys
import os
from os import listdir
from os.path import isfile, join


#f = open("./full_batch/D11-1059word.docx.txt", "r")
#paper_full =f.read()
#indis = paper_full.find("Conclusion")
#cut_end = paper_full.find("Acknowledgments")
#conclusion = paper_full[indis:cut_end]

#conclusion = "el titulo\njoeresmuyayonosequecontartepixa\n\nvamo\na\n\n\nver\nxaba\nTable 4 quenoteente\nra\nsae\n"
#conclusion = "el titulo\nTable 4 quenoteente\n"


def findFigureIndex(conclusion_doubt):

    index = "All"
    index_n = 0
    table = 0

    while table < 3:

        doubt_temp = conclusion_doubt.find("\n")

        if doubt_temp == -1:
            index = "None"
            return index
            break

        if doubt_temp < 10:
            table += 1
        else:
            table = 0
            index = "None"

        if table == 3:
            return index
            break

        if table == 1:
            endline = conclusion_doubt[doubt_temp:].find("\n")
            index = conclusion_doubt[doubt_temp:]

        conclusion_doubt2 = conclusion_doubt[doubt_temp+1:]
        conclusion_doubt = conclusion_doubt2



def deleteFigure(index, conclusion):

    index_n = conclusion.find(index)
    temp = conclusion[index_n:]
    table_i = temp.find("Table ")
    temp2 = temp[table_i:]
    temp1 = conclusion[:index_n]
    conclusion = temp1 + " " + temp2

    return conclusion



def figureHunter(conclusion):

    paper_cut = conclusion.find("\n")
    conclusion_doubt = conclusion[paper_cut:]

    index = "All"

    while index is not "None":
        index = findFigureIndex(conclusion_doubt)
        conclusion_doubt = deleteFigure(index, conclusion_doubt)

    return conclusion_doubt
