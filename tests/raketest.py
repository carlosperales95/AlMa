import sys
sys.path.append('../RAKE-tutorial')
import rake
import operator

rake_object = rake.Rake("../RAKE-tutorial/data/stoplists/SmartStoplist.txt", 5, 3, 4)

sample_file = open("./full_batch/C10-1070word.docx.txt", 'r')
text = sample_file.read()
keywords = rake_object.run(text)
print "Keywords:", keywords
