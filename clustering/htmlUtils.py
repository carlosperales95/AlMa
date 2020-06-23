import string
import spacy
import json


import sys
import os

from os import listdir
from os.path import isfile, join




def fill_titles(full_content):

    if full_content == "":
        f = open('./output_template.html','r')
        template = f.read()
        f.close()
    else:
        template=full_content

    start_id = template.find('name="paper_titles">')+len('name="paper_titles">')
    #print(start_id)
    end_id = template[start_id:].find("</div>")
    start = template[:start_id]
    end = template[start_id+end_id:]

    fp = open('./titles.txt', "r")
    file = fp.read()
    message = "\n"
    count=0
    finished=False
    titles=[]
    while finished==False:
        title_start = file.find("- ")
        filename_start = file.find("_TXT/")+len("_TXT/")
        filename_end = file.find(".txt")
        if title_start != -1:
            end_title = file.find("#")
            end_sect = file.find("---------------------")
            if len(file[title_start:end_title]) > 3:
                message = message + "\t\t\t\t\t\t\t<p> <strong>("+ str(count+1) +")</strong> "+ (file[title_start:end_title].replace("\n\r", " ")) + "\t\t\t\t\t\t\t</p>" + "\n"
                titles.append([file[filename_start:filename_end], (file[title_start:end_title].replace("\n\r", " "))])
                count+=1


            if file[end_sect:].find("\n") != -1:
                new_file_id = file[end_sect:].find("\n")+1
                file = file[end_sect + new_file_id:]

            else:
                finished = True
        else:
            finished = True

    fp.close()

    content = start+message+end
    return content, titles


def fill_clustering(arg_name, full_content, mentions, s_labels, titles):

        template=full_content

        start_id = template.find('name="'+arg_name+'">')+len('name="'+arg_name+'">')
        #print(start_id)
        end_id = template[start_id:].find("</div>")
        start = template[:start_id]
        end = template[start_id+end_id:]

        label_colors = ['#2AB0E9', '#D2CA0D', '#D7665E', '#2BAF74',
                        '#CCCCCC', '#522A64', '#A3DB05', '#FC6514',
                        '#FF7F50', '#BDB76B', '#FF7F50', '#00FA9A',
                        '#FFA07A', '#FFFACD', '#006400', '#32CD32',
                        '#DC143C', '#FFEFD5', '#8FBC8F', '#808000'
                        ]

        fp = open('./outs/'+arg_name+'.txt', "r")
        line = fp.readline()
        message = "\n"
        count=0
        f = open('./comparison.txt', "w")
        msg="\n"
        sent_idxed=[]
        while line:
            line = fp.readline()

            if line.find("---") != -1 or line.find("....") != -1:
                continue
            elif line.find("Sentences in cluster n") != -1:
                count+=1
                if count > 1:
                    sent_idxed.sort(key=lambda tup: tup[0])
                    for si in sent_idxed:
                        message = message + si[1]
                    message = message + "\t\t\t\t\t</div>" + "\n"

                message = message + '\t\t\t\t\t\t<button type="button" style="background-color:'+label_colors[count-1]+' !important;" class="collapsible">' + line[line.find("Sentences in cluster n"):line.find("Sentences in cluster n")+len("Sentences in cluster n")+1] + "</button>" + "\n"
                message = message + '\t\t\t\t\t\t<div class="content">' + "\n"
                sent_idxed=[]

            else:
                paper_num=0
                if len(line) > 3:
                    found = False
                    ####while found == False:
                    for label in s_labels:

                        if line.find(label[1][:-5]) != -1 and len(line) - len(label[1][:-5]) < 10:
                            f.write("\n")
                            f.write("\n")
                            f.write(line)
                            f.write(label[1])
                            for idx, title in enumerate(titles):
                                if label[0].find(title[0]) != -1:
                                    ###print("tHE TITLE: "+title[0])
                                    paper_num = idx+1
                                    found = True
                                    break
                    if found == False:
                        paper_num = 0
                        break
                    ###print("NEXT SENT")
                    for m in mentions:
                        if isinstance(m, list):
                            men = ""
                            for m_s in m:
                                men = men+" "+m_s
                        else:
                            #print(m)
                            men = m

                        idmen=line.find(men)
                        if idmen != -1:
                            line_bef = line[:idmen]
                            line_aft = line[idmen+len(men):]
                            mention = "<strong>" + men + "</strong>"
                            line=line_bef+mention+line_aft


                    htmline = "\t\t\t\t\t\t\t<p> <strong> (Paper "+ str(paper_num) +") </strong>"+ line + "\t\t\t\t\t\t\t</p>" + "\n"
                    htmline = htmline + "\t\t\t\t\t\t\t<p>  \t\t\t\t\t\t\t</p>" + "\n"
                    sent_idxed.append([paper_num, htmline])

        f.close()
        fp.close()
        sent_idxed.sort(key=lambda tup: tup[0])
        for si in sent_idxed:
            message = message + si[1]
        message = message + "\t\t\t\t\t</div>" + "\n"

        content=start+message+end

        return content


def fill_mentions(full_content, mentions, name):

    template = full_content

    start_id = template.find('name="' + name + '">')+len('name="' + name + '">')
    #print(start_id)
    end_id = template[start_id:].find("</div>")
    start = template[:start_id]
    end = template[start_id+end_id:]

    message=""
    for m in mentions:
        if isinstance(m, list):
            men = ""
            for m_s in m:
                men = men+" "+m_s
        else:
            #print(m)
            men = m

        message = message + "\t\t\t\t\t\t\t<p> "+ men + "\t\t\t\t\t\t\t</p>" + "\n"
        message = message + "\t\t\t\t\t\t\t<p>  \t\t\t\t\t\t\t</p>" + "\n"


    content = start+message+end
    return content


def fill_menperPaper(full_content, mentions):

    template = full_content

    start_id = template.find('name="papers-mentions">')+len('name="papers-mentions">')
    #print(start_id)
    end_id = template[start_id:].find("</div>")
    start = template[:start_id]
    end = template[start_id+end_id:]


    dir = "./sum_batch/"

    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

    texts=[]
    for f in onlyfiles:
        file = open(dir+f, 'r')
        t = file.read()
        texts.append(t)


    message = "\n"
    for idx, paper in enumerate(texts):
        found_men=[]
        message = message + '\t\t\t\t\t\t\t\t<button type="button" id="subsubcollapsible" class="collapsible"> Paper ('+ str(idx +1) +') </button>\n'
        message = message + '\t\t\t\t\t\t\t\t\t<div class="content"> \n'

        for m in mentions:
            f_m=""
            if isinstance(m, list):
                for p in m:
                    f_m = f_m + " " + p
            else:
                f_m = m
            if paper.find(f_m) != -1:
                found_men.append(f_m)

        concat = ""
        for f in found_men:
            concat=concat+f+", "
        message = message + '\t\t\t\t\t\t\t\t\t\t<p>' + concat + '</p> \n'
        message = message + '\t\t\t\t\t\t\t\t\t</div> \n'


    content = start+message+end
    return content


def dynamicfill_output(mentions, c_labels, e_labels):

    full_content = ""
    full_content, titles = fill_titles(full_content)

    full_content = fill_clustering("bigram_claims", full_content, mentions, c_labels, titles)
    full_content = fill_clustering("trigram_claims", full_content, mentions, c_labels, titles)
    full_content = fill_clustering("bigram_evidences", full_content, mentions, e_labels, titles)
    full_content = fill_clustering("trigram_evidences", full_content, mentions, e_labels, titles)


    right_m=[]
    center_m=[]
    left_m=[]
    order=1
    for m in mentions:
        if order == 1:
            left_m.append(m)
        if order == 2:
            center_m.append(m)
        if order == 3:
            right_m.append(m)
            order = 0

        order+=1

    full_content = fill_mentions(full_content, left_m, "left-mentions")
    full_content = fill_mentions(full_content, center_m, "center-mentions")
    full_content = fill_mentions(full_content, right_m, "right-mentions")
    full_content = fill_menperPaper(full_content, mentions)

    f = open('./outs/batch_statistics_view.html', "w")
    f.write(full_content)
    f.close()


#    message+message+msg
#    f.write(message)
    #f.close()

    return None
