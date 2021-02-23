import urllib.request
from bs4 import BeautifulSoup
import re
import csv
import os
import json
import requests
from requests.exceptions import RequestException
import pandas as pd
import joblib
from underthesea import word_tokenize
import numpy as np
from constants import REQUEST_HEADERS
from utils import StringUtils

def get_comment_from_url(url):
    try:
        if url:
            response = requests.get(url, timeout=60, headers=REQUEST_HEADERS)
            if response.status_code != 200:
                print('Error. can not get content from {url}')
                return None
            source_content = response.content
            soup_content = BeautifulSoup(source_content, "html.parser")
            script = soup_content.find_all("script", attrs={"type": "application/ld+json"})[0]
            script = str(script)
            script = script.replace("</script>","").replace("<script type=\"application/ld+json\">","")

            csvdata = []

            for element in json.loads(script)["review"]:
                if "reviewBody" in element:
                    csvdata.append([element["reviewBody"]])
            if csvdata:
                return csvdata
        return None
    except RequestException as ex:
        return None


def standardize_data(row):
    # remove all special charactor
    row = re.sub(r'[^a-zA-Z0-9 ]',r'', row)
    row = row.strip()
    return row

# Tokenizer
def tokenizer(row):
    return word_tokenize(row, format="text")

def analyze(result):
    bad = np.count_nonzero(result)
    good = len(result) - bad
    print("Number of bad comments = ", bad)
    print("Number of good comments = ", good)

    if good>bad:
        return "Okay! You can buy it!"
    else:
        return "Oh no! Please check it carefully!"

# 1. Load URL and print comments
url = input('Enter your url:')
if not url:
    url = "https://www.lazada.vn/products/quan-boi-nam-hot-trend-i244541570-s313421582.html?spm=a2o4n.searchlist.list.11.515c365foL7kyZ&search=1"
data = get_comment_from_url(url)

# 2. Standardize data
data_frame = pd.DataFrame(data)
data_frame[0] = data_frame[0].apply(standardize_data)

# 3. Tokenizer
data_frame[0] = data_frame[0].apply(tokenizer)

# 4. Embedding
X_val = data_frame[0]
emb = joblib.load('tfidf.pkl')
X_val = emb.transform(X_val)
print(X_val)

# 5. Predict
model = joblib.load('saved_model.pkl')
result = model.predict(X_val)
print('11111', result)
print(analyze(result))
print("Done")




