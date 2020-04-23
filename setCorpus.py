import sys
import json
import os
from os import listdir
from os.path import isfile, join
import random
from subprocess import Popen
import shutil



onlyfiles = [f for f in listdir("./sum_batch/") if isfile(join("./sum_batch/", f))]

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

print("New batch structure created in ./rank/batch_" +  str(num) + "/\n")


print("Executing MARGOT on papers.....")

for idx,file in enumerate(onlyfiles):
    print("MARGOT for file: " + file[:-4] + "......... (" + str(idx+1) + "/" + str(len(onlyfiles)) + ")\n" )

    process = Popen(['../MARGOT/run_margot.sh ' + "../AlMa/sum_batch/" + file + " " + "output"], shell=True, cwd="../MARGOT/")
    process.wait()

    newPath = shutil.copy('../MARGOT/output/OUTPUT.json', './rank/batch_' + str(num) + '/MARGOT_output/' + file[:-4] + '/')
    print("\nMoving OUTPUT.json to batch structure...")
    print("\n")
