from bs4 import BeautifulSoup
import re
import csv
import os
import json
import string
import requests
from requests.exceptions import RequestException
import pandas as pd
import joblib
from underthesea import word_tokenize
import numpy as np
from constants import REQUEST_HEADERS


def get_comment_from_url(url):
    try:
        if url:
            response = requests.get(url, timeout=60, headers=REQUEST_HEADERS)
            if response.status_code != 200:
                print('Error. can not get content from {url}')
                return None
            source_content = response.content
            soup_content = BeautifulSoup(source_content, "html.parser")
            content = soup_content.find_all("script", attrs={"type": "application/ld+json"})[0]
            content = str(content)
            content = content.replace("</script>","").replace("<script type=\"application/ld+json\">","")

            csvdata = []

            for element in json.loads(content)["review"]:
                if "reviewBody" in element:
                    csvdata.append([element["reviewBody"]])
            if csvdata:
                return csvdata
        return None
    except RequestException as ex:
        return None


def standardize_data(row):
    # remove all special charactor
    row = re.sub(r"[-()#/@;:<>{}`+=~|.!?,]", "", row)
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
    url = "https://www.lazada.vn/products/quan-dui-the-thao-nam-nhieu-mau-du-size-vai-xi-gian-nhe-min-mat-i246614289-s1287480682.html"
data = get_comment_from_url(url)
print('data crawl:', data)

# 2. Standardize data
data_frame = pd.DataFrame(data)
data_frame[0] = data_frame[0].apply(standardize_data)
print('Standardize data:', data_frame)

# 3. Tokenizer
data_frame[0] = data_frame[0].apply(tokenizer)
print('tokenizer:', data_frame)

# 4. Embedding
X_val = data_frame[0]
emb = joblib.load('tfidf.pkl')
X_val = emb.transform(X_val)
print('Embedding:', X_val)

# 5. Predict
model = joblib.load('saved_model.pkl')
result = model.predict(X_val)
print('Predict:', analyze(result))
