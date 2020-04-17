import sys
import json
import os
from os import listdir
from os.path import isfile, join
import random





if len(sys.argv) == 1:
    print("\n")
    print("Directory to batch not specified.")
    print("Call should be directed to a specific batch.")
    print("Try again.")
    sys.exit()


batch_path = str(sys.argv[1])


def list_paths(path):
    directories = [x[1] for x in os.walk(path)]
    non_empty_dirs = [x for x in directories if x] # filter out empty lists
    return [item for subitem in non_empty_dirs for item in subitem] # flatten the list

dirs = list_paths(batch_path + "MARGOT_output/")


for idx,dir in enumerate(dirs):

    input_file = open (batch_path + "MARGOT_output/" + dir + "/OUTPUT.json")

    imported_json = json.load(input_file)

    evidences_json = []
    claims_json = []


    for x in imported_json:
        for y in imported_json[x]:
            if y["evidence_score"] >= 0:
                evidences_json.append(y)
            if y["claim_score"]  >= 0:
                claims_json.append(y)


    def sorted_ev_score(evidences_json):
        try:
            return float(evidences_json["evidence_score"])
        except KeyError:
            return 0

    def sorted_claim_score(claims_json):
        try:
            return float(claims_json["claim_score"])
        except KeyError:
            return 0


    evidences_json.sort(key=sorted_ev_score, reverse=True)
    claims_json.sort(key=sorted_claim_score, reverse=True)


    f = open(batch_path + "evidences_" + dir +".json", "w")
    f.write(json.dumps(evidences_json, indent=4, sort_keys=True))
    f.close()


    f = open(batch_path + "claims_" + dir + ".json", "w")
    f.write(json.dumps(claims_json, indent=4, sort_keys=True))
    f.close()
