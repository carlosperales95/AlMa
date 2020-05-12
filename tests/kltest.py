import kleis.resources.dataset as kl
default_corpus = kl.load_corpus()

# Default: filter_min_count = 3
# Default: tagging_notation="BILOU"
default_corpus.training(features_method="simple-posseq", filter_min_count=10)

#sample_file = open("./sum_batch/sum_C10-1070word.docx.txt", 'r')
sample_file = open("./sum_batch/sum_C10-1070word.docx.txt", 'r')

text = sample_file.read()

keyphrases = default_corpus.label_text(text)



# Print result
print("Keyphrases:", len(keyphrases))

# Each keyphrase has the fields needed for the brat format
print("\n".join([str(k) for k in keyphrases]))

# The fields are
keyphrase_id, (keyphrase_label, (keyphrase_start, keyphrase_end)), keyphrase_text = keyphrases[0]

print("\n - Fields: ", keyphrase_id, (keyphrase_label, (keyphrase_start, keyphrase_end)), keyphrase_text)

# (keyphrase_start, keyphrase_end) are the span in the original text
print("\n Segment of text: '%s'" % text[keyphrase_start:keyphrase_end])
print(text[keyphrase_start:keyphrase_end] == keyphrase_text)


# Print keyphrases in brat format
print(kl.keyphrases2brat(keyphrases))
sfile = open("./klkeywords.txt", 'w')

# print label, start and end
for keyphrase in keyphrases:

    sfile.write("- - - - - KEYPHRASE - - - - -\n")
    sfile.write("Label: " + kl.keyphrase_label(keyphrase) + "\n")
    sfile.write("Span: " + str(kl.keyphrase_span(keyphrase)) + "\n")
    # Another example with span
    start, end = kl.keyphrase_span(keyphrase)
    sfile.write("Start: " + str(start) + "\n")
    sfile.write("End: " + str(end) + "\n")
    # Without a function
    keyphrase_id, (keyphrase_label, (start, end)), keyphrase_str = keyphrase
    sfile.write("All fields: " + str(keyphrase_id) +  " " + keyphrase_label + " " + str(start) + " " + str(end) + " " + keyphrase_str + "\n\n")


sfile.close()
