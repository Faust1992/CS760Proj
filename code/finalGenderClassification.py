import random
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import re
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse.csr import csr_matrix
from scipy.sparse import coo_matrix, vstack

twigen = pd.read_csv("louis_tweets_1127_labeled.csv", encoding='ISO-8859-1')
twigen.head()


def normalize_text(s):
    s = str(s)
    s = s.lower()
    s = re.sub('http://.*', "", s)
    s = re.sub('\@.*?\ ', "", s)
    s = re.sub('\#', "", s)
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)
    s = re.sub('\s+', ' ', s)
    s = re.sub('http://.*', "", s)
    s = re.sub('\@.*?\ ', "", s)
    return s


twigen['text_norm'] = [normalize_text(s) for s in twigen['content']]
twigen['gender'].replace('F', 'female', inplace=True)
twigen['gender'].replace('M', 'male', inplace=True)
twigen['gender'].replace('O', 'brand', inplace=True)
twigen['gender'] = twigen['gender'].fillna("")
new1 = []
new2 = []
s = ""
id = twigen['user_id'].get(0)
gender = twigen['gender'].get(0)
for i in range(len(twigen['user_id'])):
    if (twigen['user_id'][i] == id):
        s += twigen['text_norm'][i]
    else:
        new1.append(s)
        new2.append(gender)
        s = ""
        id = twigen['user_id'][i]
        gender = twigen['gender'][i]
new1.append(s)
new2.append(gender)
newtwigen = pd.Series(new1)
newgender = pd.Series(new2)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(newtwigen)
encoder = LabelEncoder()
y = encoder.fit_transform(newgender)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.04837431)
nb = MultinomialNB();
#cv = ShuffleSplit(n_splits=2, test_size=0.5, random_state=0)
#scores = cross_val_score(nb, x, y, cv=cv)  # , scoring='f1_macro')
nb.fit(x_train, y_train)
#print(scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
count = 0
testnum=[]
for i in range(122):
    P = nb.predict_proba(x_test[i])[0]
    index1 = y_test[i]
    index2 = np.argmax(P)
    if index1 == index2:
        count+= 1
    num = x.toarray().tolist().index(x_test.toarray()[i].tolist())
    testnum.append(num)
    print(num)
    print(P)
for i in range(len(new1)):
    if i not in testnum:
        P = nb.predict_proba(x[i])[0]
        print(i)
        print(P)
print(float(count)/122.0)