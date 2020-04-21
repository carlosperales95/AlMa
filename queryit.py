import urllib.parse
import urllib.request
from bs4 import BeautifulSoup
import re


g_clean = [ ]
url = 'http://aipano.cse.ust.hk/p13/#'
values = { 'query': '' }

data = urllib.parse.urlencode(values)
url = '?'.join([url, data])
response = urllib.request.urlopen(url)

the_page = response.read()

soup = BeautifulSoup(the_page, 'lxml')
a = soup.find_all('a') # a is a list
print(a)
for i in a:
    k = i.get('href')
    try:
        m = re.search("(?P<url>https?://[^\s]+)", k)
        n = m.group(0)
        rul = n.split('&')[0]
        domain = urlparse(rul)
        g_clean.append(rul)
    except:
        continue

print (g_clean)
print(the_page)
