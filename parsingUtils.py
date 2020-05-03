import sys
import os
from os import listdir
from os.path import isfile, join
import shutil




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



def cleanLineJumps(paper_title):

    res = [i for i in range(len(paper_title)) if paper_title.startswith("\n", i)]
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
    return paper_title



def findTitle(paper):


    index_title = paper.find("Abstract")
    if index_title == -1:
        index_title = paper.find("ABSTRACT")

    if index_title == -1:
        index_title = paper.find("Introduction")

    if index_title == -1:
        index_title = paper.find("INTRODUCTION")

    paper_temp = paper[index_title:]
    find_index_2 = index_title + paper_temp.find('\n')

    paper_title = paper[:find_index_2]

    paper_cut_start = paper_title.find("\n")
    paper_title2 = ""
    while paper_cut_start < 2:
            paper_cut_start = 1
            paper_title2 = paper_title[paper_cut_start:]
            paper_title = paper_title2
            paper_cut_start = paper_title.find('\n')

            if paper_cut_start == -1:
                break


    paper_title = paper_title[0:]
    paper_cut = paper_title.find("\n")

    paper_doubt = paper_title[paper_cut:]
    paper_doubt_cut = 0

    block = 0
    init_block = "empty"

    while block < 2:

        paper_doubt_temp = paper_doubt.find("\n")

        if paper_doubt_temp > 1:
            block += 1

        else:
            block = 0
            init_block = paper_doubt[paper_doubt_temp:]

        if paper_doubt_temp == -1:
            break

        paper_doubt2 = paper_doubt[paper_doubt_temp+1:]
        paper_doubt = paper_doubt2


    index = paper_title.find(init_block)
    paper_title =  cleanLineJumps(paper_title[:index])

    return paper_title



def getAbstract(paper_full):


    find_index_safe = paper_full.find('Introduction')
    if find_index_safe == -1:
        find_index_safe = paper_full.find("INTRODUCTION")

    find_index = paper_full.find('Abstract')
    if find_index == -1:
        find_index = paper_full.find("ABSTRACT")

    if find_index == -1:
        find_index = paper_full.find("Introduction")

    if find_index == -1:
        find_index = paper_full.find("INTRODUCTION")


    paper_temp = paper_full[find_index:]
    find_index_2 = find_index + paper_temp.find('\n')

    line_abstract = paper_full[find_index:find_index_2] + "\n"

    paper_temp = paper_full[find_index_2:]
    if find_index_safe > find_index:
        find_index = find_index_2 + paper_temp.find('Introduction') - 6
        if paper_temp.find('Introduction') == -1:
            find_index = find_index_2 + paper_temp.find('INTRODUCTION') - 6
            #12

    else:
        find_index = find_index_2 + paper_temp.find('\n\n2. ')
        if paper_temp.find('\n\n2. ') == -1:
            find_index = find_index_2 + paper_temp.find('\n\n2 ')
            if paper_temp.find('\n\n2 ') == -1:
                find_index = find_index_2 + paper_temp.find('\n2 ')

    abstract = paper_full[find_index_2:find_index].replace("\n", "")
    abstract = abstract.replace('\n', ' ').replace('\r', ' ').replace('   ', ' ').replace('- ', '').replace('  ', ' ')
    abstract = line_abstract + abstract

    return abstract



def getConclusion(paper_full):


    find_index = paper_full.find('Conclusion')

    if find_index == -1:
        find_index = paper_full.find('Discussion')

    paper_temp = paper_full[find_index:]
    find_index_2 = find_index + paper_temp.find('\n')

    find_index = paper_temp.find('Acknowledgement')
    if find_index == -1:
        find_index = paper_temp.find('Acknowledgment')
        if find_index == -1:
            find_index = paper_full.find('Bibliography')
            if find_index == -1:
                find_index = paper_temp.find('References')


    temp_conclusion = paper_full[find_index_2 : (find_index_2 + find_index)]

    return temp_conclusion



def getIntro(paper_full):


    find_index = paper_full.find("Introduction")

    if find_index == -1:
        find_index = paper_full.find("INTRODUCTION")


    paper_temp = paper_full[find_index:]
    find_index_2 = find_index + paper_temp.find('\n')

    line_intro = paper_full[find_index:find_index_2] + "\n"

    print(line_intro)
    find_index = 1000
    while find_index > 3000:
        find_index = paper_temp.find('\r\n2. ')
        if paper_temp.find('\r\n2. ') == -1:
            find_index = paper_temp.find('\n\n2. ')
            if paper_temp.find('\n\n2. ') == -1:
                find_index = paper_temp.find('\n\n2 ')
                if paper_temp.find('\n\n2 ') == -1:
                    break

    print(find_index)

    intro = paper_temp[:find_index]
    print(paper_temp[:find_index])
    intro = intro.replace('\n', ' ').replace('\r', ' ').replace('   ', ' ').replace('- ', '').replace('  ', ' ')
    intro = line_intro + intro

    return intro