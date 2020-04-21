import requests
image_url = "http://aipano.cse.ust.hk/p13/#"
pdf_url = "https://arxiv.org/pdf/1301.3781.pdf"

# URL of the image to be downloaded is defined as image_url
r = requests.get(pdf_url) # create HTTP response object

# send a HTTP request to the server and save
# the HTTP response in a response object called r
print(r.content)
