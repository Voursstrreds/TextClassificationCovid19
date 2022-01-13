import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from pprint import pprint

import csv
import time

from statics import *


def calculate_class_and_prob(training_data):

    columns = ['sent', 'class']
    rows = []

    new_sentence = 'what is the price of the book well '
    new_word_list = word_tokenize(new_sentence)

    training_data = pd.DataFrame(training_data[0], columns=columns)

    class_names = training_data['class'].unique()

    class_freqs = training_data.groupby('class').count() / training_data.shape[0]

    docs = [row['sent'] for index, row in training_data.iterrows()]

    vec = CountVectorizer()
    x = vec.fit_transform(docs)

    total_features = len(vec.get_feature_names_out())

    tdms = []
    freqs = []
    probs = []
    answers = []
    newWords = []

    for idx, label in enumerate(class_names):

        class_docs = [row['sent'] for index, row in training_data.iterrows() if row['class'] == label]

        vec_s = CountVectorizer()
        x_s = vec_s.fit_transform(class_docs)
        tdms.append(pd.DataFrame(x_s.toarray(), columns=vec_s.get_feature_names_out()))

        word_list_s = vec_s.get_feature_names_out()
        count_list_s = x_s.toarray().sum(axis=0)

        freqs.append(dict(zip(word_list_s, count_list_s)))

        prob = []

        for word, count in zip(word_list_s, count_list_s):
            prob.append(count / len(word_list_s))
        probs.append(dict(zip(word_list_s, prob)))

        total_cnts_features_s = count_list_s.sum(axis=0)

        prob_s_with_ls = []
        for word in new_word_list:
            if word in freqs[-1].keys():
                count = freqs[-1][word]
            else:
                count = 0
            prob_s_with_ls.append((count + 1) / (total_cnts_features_s + total_features))
        newWords.append(dict(zip(new_word_list, prob_s_with_ls)))

        answer = 1
        for word in new_word_list:
            answer = answer * newWords[-1][word]
            # print(word, newWords[-1][word])
        try:
            answers.append(answer * class_freqs['sent'][idx + 1])
        except KeyError:
            answers.append(0)

    return tdms, probs, answers


def main():
    training_data, testing_data = get_train_and_test_data()
    tdms, probs, answers = calculate_class_and_prob(training_data)

    pprint(probs)


main()

'''

q_docs = [row['sent'] for index,row in training_data.iterrows() if row['class'] == 'question']

vec_q = CountVectorizer()
X_q = vec_q.fit_transform(q_docs)
tdm_q = pd.DataFrame(X_q.toarray(), columns=vec_q.get_feature_names())

#pp.pprint(tdm_q)



word_list_s = vec_s.get_feature_names();    
count_list_s = X_s.toarray().sum(axis=0) 
freq_s = dict(zip(word_list_s,count_list_s))

#pp.pprint(freq_s)

word_list_q = vec_q.get_feature_names();    
count_list_q = X_q.toarray().sum(axis=0) 
freq_q = dict(zip(word_list_q,count_list_q))
#pp.pprint(freq_q)

prob_s=[]

for word,count in zip(word_list_s,count_list_s):
    prob_s.append(count/len(word_list_s))
ProbS=dict(zip(word_list_s,prob_s))

#pp.pprint(ProbS)

prob_q=[]
for word,count in zip(word_list_q,count_list_q):
    prob_q.append(count/len(word_list_q))
ProbQ=dict(zip(word_list_q,prob_q))

#pp.pprint(ProbQ)

total_cnts_features_s = count_list_s.sum(axis=0)
total_cnts_features_q = count_list_q.sum(axis=0)

#pp.pprint(total_cnts_features_s)
#pp.pprint(total_cnts_features_q)


prob_s_with_ls = []
for word in new_word_list:
    if word in freq_s.keys():
        count = freq_s[word]
    else:
        count = 0
    prob_s_with_ls.append((count + 1)/(total_cnts_features_s + total_features))
newWordsS=dict(zip(new_word_list,prob_s_with_ls))

#pp.pprint(newWordsS)


answerStatement = 1

# run a loop
for i in newWordsS:
    answerStatement = answerStatement*newWordsS[i]

#pp.pprint(answerStatement)

prob_q_with_ls = []
for word in new_word_list:
    if word in freq_q.keys():
        count = freq_q[word]
    else:
        count = 0
    prob_q_with_ls.append((count + 1)/(total_cnts_features_q + total_features))
newWordsQ=dict(zip(new_word_list,prob_q_with_ls))

answerQuestion = 1

# run a loop
for i in newWordsQ:
    answerQuestion = answerQuestion*newWordsQ[i]

#pp.pprint(newWordsQ)

#pp.pprint(answerQuestion)

'''