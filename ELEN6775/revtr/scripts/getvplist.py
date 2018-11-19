#!/Users/alexstein/anaconda3/bin/python

import re, pycurl
from io import BytesIO
from bs4 import BeautifulSoup

url = "http://bgoodc.cs.columbia.edu/rr/measurements_2017/"

# the python wget package has no options, so using curl instead
def cget(url):
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    # TODO: put the username and password in a file
    c.setopt(c.USERPWD, 'rr:ELEN6775ResearchGroup')
    c.setopt(c.WRITEDATA, buffer)
    c.perform()
    c.close()
    return buffer.getvalue().decode('iso-8859-1')

# get the list of vp-csv files
vplist = set()
vplisthtml = cget(url)
soup = BeautifulSoup(vplisthtml,'html.parser')
for hit in soup.find_all('a'):
    match = re.match(r'^.*csv', hit['href'])
    if match:
        vplist.add(match[0])

# for each CSV file in the list, process it!
print("\n".join(vplist))

