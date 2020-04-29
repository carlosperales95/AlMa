import sys
import os
from os import listdir
from os.path import isfile, join
import shutil

#f = open("./full_batch/D11-1059word.docx.txt", "r")
#paper =f.read()
#indis = paper.find("Conclusion")
#cut_end = paper.find("Acknowledgments")
#conclusion = paper[indis:cut_end]

#conclusion = "el titulo\njoeresmuyayonosequecontartepixa\n\nvamo\na\n\n\nver\nxaba\nTable 4 quenoteente\nra\nsae\n"
#conclusion = "el titulo\nTable 4 quenoteente\n"



def cleanBatch(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)



def scisummFindTXT(path, txt_folder_name):

    txt_dirs = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for dir in d:
            if str(dir).find(txt_folder_name) == 0:
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

    return txt_files
