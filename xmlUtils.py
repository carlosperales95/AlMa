import sys
import os
from os import listdir
from os.path import isfile, join
import shutil


def xmlToSections(file):

    topic_sections_titles = []
    
    if os.path.isfile(file):
        f = open(file, "r")
        xmlstring = f.read()
        while xmlstring.find('<SECTION ') != -1:

            index = xmlstring.find('<SECTION ')
            temp_xmlstring = xmlstring[index+9:]

            end_title_ind = temp_xmlstring.find("number =")
            if end_title_ind == -1 or end_title_ind > 70:
                end_title_ind = temp_xmlstring.find(">")

            title = temp_xmlstring[len("title="): end_title_ind]
            ind = title.find(".")
            if ind != -1:
                title = title[:ind]

            topic_sections_titles.append(title.replace('"', ""))

            next_index = temp_xmlstring.find('</SECTION>')
            xmlstring = temp_xmlstring[next_index:]
            #print(title.replace('"', ""))

        f.close()

    return topic_sections_titles


def scisummFindXML(path, xml_folder_name):

    xml_dirs = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for dir in d:
            if str(dir).find(xml_folder_name) == 0:
                xml_dirs.append(os.path.join(r, dir))


    xml_files = []
    for dir in xml_dirs:
        count = 0
        temp = []
        for r, d, f in os.walk(dir):
            for file in f:
                if file.find('.xml') != -1:
                    xml_files.append(str(dir)+"/"+str(file))

    return xml_files
