import sys
from os import listdir
from os.path import isfile, join




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
    if table_i == -1:
        table_i = temp.find("Figure ")
    if table_i == -1:
        table_i = temp.find("\r\n\r\n")


    temp2 = temp[table_i:]
    temp1 = conclusion[:index_n]
    conclusion = temp1 + " " + temp2

    return conclusion


def removeFigureTags(conclusion, name):

    while conclusion.find(name) != -1:

        index = conclusion.find(name)
        conclusion_temp = conclusion [index: ]
        index_end = conclusion_temp.find("\n")
        conclusion2 = conclusion[:index] + conclusion_temp[index_end:]
        conclusion = conclusion2

    return conclusion


def figureHunter(conclusion_doubt):

    index = "All"

    while index != "None":
        index = findFigureIndex(conclusion_doubt)
        conclusion_doubt2 = deleteFigure(index, conclusion_doubt)
        conclusion_doubt = conclusion_doubt2

    return conclusion_doubt





f = open("text.txt", "r")
conclu =f.read()

conclusion = figureHunter(conclu)
conclusion = removeFigureTags(conclusion, "Figure ")
conclusion = removeFigureTags(conclusion, "Table ")





print(conclusion)
