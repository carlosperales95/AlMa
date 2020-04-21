=== AlMa PACKAGE ===

1. Get your scientific papers in the <folder name> in .txt format


2. Run the script paperSum.py, this will summarize the papers into Abstract and Conclusion

3. Run the setCorpus.py script to prepare the batch folders and subfolders for all the papers in the ./rank directory
It will then run all your summarized papers through MARGOT and move the output to the proper subfolders in ./rank/<batch_number>/MARGOT_input/

4. Run argRank.py. It will separate claims and evidences from the aummarized papers and rank them in terms of score. These .json files can be found in ./rank/<batch_number>/

