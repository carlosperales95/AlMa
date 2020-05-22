# coding: utf-8
import sys
import os
from os import listdir
from os.path import isfile, join
import shutil
import re



def findFigureIndex(conclusion_doubt):

    index = "All"
    index_n = 0
    table = 0

    while table < 4:

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

        if table == 4:
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

    #line_abstract = paper_full[find_index:find_index_2] + "\n"

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
    abstract = abstract.replace(' \n', ' ').replace(' \r', ' ').replace('\n', '').replace('\r', '').replace('   ', ' ').replace('- ', '').replace('  ', ' ')
    #abstract = line_abstract + abstract

    return abstract



def getConclusion(paper_full):


    find_index = paper_full.find('Conclusion')
    if find_index == -1:
        find_index = paper_full.find('CONCLUSION')
        if find_index == -1:
            find_index = paper_full.find('Conclusions')
            if find_index == -1:
                find_index = paper_full.find('CONCLUSIONS')
                if find_index == -1:
                    find_index = paper_full.find('Discussion')
                    if find_index == -1:
                        find_index = paper_full.find('DISCUSSION')
                        if find_index == -1:
                            find_index = paper_full.find('Contribution')
                            if find_index == -1:
                                find_index = paper_full.find('CONTRIBUTION')
                                if find_index == -1:
                                    find_index = paper_full.find('Summary')
                                    if find_index == -1:
                                        find_index = paper_full.find('SUMMARY')

    paper_temp = paper_full[find_index:]
    find_index_2 = paper_temp.find('\n')
    paper_temp = paper_full[find_index + find_index_2 : ]


    find_index = paper_temp.find('Acknowl')
    if find_index == -1:
        find_index = paper_temp.find('ACKNOWL')
        if find_index == -1:
            find_index = paper_full.find('Bibl')
            if find_index == -1:
                find_index = paper_temp.find('BIBL')
                if find_index == -1:
                    find_index = paper_temp.find('Refer')
                    if find_index == -1:
                        find_index = paper_temp.find('REF')

    temp_conclusion = paper_temp[ :find_index]

    return temp_conclusion



def getSection(paper_full, section_name, next_section):

    find_index = paper_full.find(section_name[: len(section_name) - 1])

    if find_index != -1:
        paper_temp = paper_full[find_index:]
        find_end = paper_temp.find("\n")
        title = False
        section_temp = ""
        if next_section.endswith(' '):
            next_section = next_section[:-1]
        while title != True:
            find_next_index = paper_temp.find(next_section)
            if find_next_index != -1:
                section_temp = paper_temp[find_next_index:]
                find_end2 = section_temp.find("\n\r\n")
                if find_end2 <= 35:
                    title = True
                else:
                    paper_change = paper_temp.find(section_temp)
                    paper_temp2 = paper_temp[paper_change + find_end2:]
                    paper_temp = paper_temp2
            else:
                section = ""
                break

        section_start = paper_full.find(section_temp)
        section = paper_full[find_index + find_end : section_start]
        #section = paper_full[find_index + len(section_name) : find_next_index]
        #section = figureHunter(section)

        section = section.replace('\n', ' ').replace('\r', ' ').replace('   ', ' ').replace('- ', '').replace('  ', ' ').replace('\t', ' ')
    else:
        section = ""

    return section


def cleanTextRubble(text):
    text = re.sub(r'[\*]','',text)
    text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+','',text)
    text = re.sub(r'(/[aA-zZ,0-9,:%,\.,-]+)+\s', '', text)
    text = re.sub(r'\([a-k]\)','',text)
    text = re.sub(r'[[a-k]\]','',text)
    text = re.sub(r'-?\s\s(\s)*','',text)
    text = re.sub(r'[0-9].?,?([0-9])(\s[0-9].?,?[0-9] )+','',text)
    text = re.sub(r'[\∗]','',text)
    text = re.sub(r'\s[0-9]\)\s', ') ',text)
    text = re.sub(r'\s[a-z]\)\s', ') ',text)
    text = re.sub(r'\s\)\s', ' ',text)
    text = re.sub(r'\;\)\s', '; ',text)
    text = re.sub(r'\.\.','.',text)
    text = re.sub(r'\.\s\s[0-9]\s', '. ', text)
    text = re.sub(r'\.\s(\s)*\.','',text)
    text = re.sub(r'\([[0-9]([0-9]+)?\)','',text)
    text = re.sub(r'\[[0-9]([0-9]+)?\]','',text)
    text = re.sub(r'\.[\s]?([aA-zZ,0-9]+\s)+\n','',text)

    return text
