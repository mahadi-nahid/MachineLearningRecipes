# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 13:17:13 2016

@author: NAHID
"""

import os
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score

NEWLINE = '\n'

HAM = 'ham'
SPAM = 'spam'

#SOURCES = [
#    ('data/spam',        SPAM),
#    ('data/easy_ham',    HAM),
#    ('data/hard_ham',    HAM),
#    ('data/beck-s',      HAM),
#    ('data/farmer-d',    HAM),
#    ('data/kaminski-v',  HAM),
#    ('data/kitchen-l',   HAM),
#    ('data/lokay-m',     HAM),
#    ('data/williams-w3', HAM),
#    ('data/BG',          SPAM),
#    ('data/GP',          SPAM),
#    ('data/SH',          SPAM)
#]


SOURCES = [
    ('data/spam',        SPAM),
    ('data/ham',    HAM)
]

SKIP_FILES = {'cmds'}


def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    past_header, lines = False, []
                    f = open(file_path, encoding="latin-1")
                    for line in f:
                        if past_header:
                            lines.append(line)
                        elif line == NEWLINE:
                            past_header = True
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content


def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))

data = data.reindex(numpy.random.permutation(data.index))

pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
    ('classifier',         MultinomialNB())
])

k_fold = KFold(n=len(data), n_folds=6)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values.astype(str)

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values.astype(str)

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label=SPAM)
    scores.append(score)

print('Total emails classified:', len(data))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)