=== AlMa PACKAGE ===

1. Run the script searchByTopic.py and add the topic words of choose as an argument.
It will run through the scientific papers in the database, search for the topic words in the title and get them ready for later processing in the folder ./full_batch.

2. Run the script paperSum.py, this will summarize the papers into Abstract and Conclusion

3. Run the setCorpus.py script to prepare the batch folders and subfolders for all the papers in the ./rank directory
It will then run all your summarized papers through MARGOT and move the output to the proper subfolders in ./rank/<batch_number>/MARGOT_input/

4. Run argRank.py. It will separate claims and evidences from the summarized papers and rank them in terms of score. These .json files can be found in ./rank/<batch_number>/

5. Run phrase2vec.py in the ./clustering folder with ./rank/batch<batch_number> argument.
This way, it will process the arguments from the batch you choose.

6. Open the file .outs/batch_statistics_view.html in your browser. It will show all the important results from the AlMa process
