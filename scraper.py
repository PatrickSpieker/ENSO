

import urllib
import bs4

url = "https://www1.ncdc.noaa.gov/pub/data/cmb/ersst/v4/netcdf/"
urllib.urlretrieve(url, "idx.html")
soup = bs4.BeautifulSoup(open("idx.html"), 'lxml')

k = 0
for i in soup.find_all("a"):
    if "ersst.v4" in i["href"]:
        urllib.urlretrieve(url + i["href"], i["href"])
        print i['href']  + ": yay"
    else: 
        print i["href"] + ": boo"
