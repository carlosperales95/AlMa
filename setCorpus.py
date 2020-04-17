import sys
import json
import os
from os import listdir
from os.path import isfile, join
import random


onlyfiles = [f for f in listdir("./sum_corpus/") if isfile(join("./sum_corpus/", f))]

num = 0
try:
        num = random.randint(1, 9999)
        os.mkdir("./rank/batch_" + str(num) + "/")
        os.mkdir("./rank/batch_" + str(num) + "/MARGOT_output/")
except:
        num = random.randint(1, 9999)
        os.mkdir("./rank/batch_" + str(num) + "/")
        os.mkdir("./rank/batch_" + str(num) + "/MARGOT_output/")


print("Creating dir structure for Batch " + str(num) + "....")
print("\n")

for idx,file in enumerate(onlyfiles):
    try:
            os.mkdir("./rank/batch_" + str(num) + "/MARGOT_output/" + file[:-4] + "/")
    except:
            print("Skip -- Subfolder already exists: ./rank/batch_" + str(num) + "/MARGOT_output/" + file[:-4] + "/\n")

print("New batch structure created in ./rank/batch_" +  str(num) + "/")
