import numpy as np
import pandas as pd
import pprint as pp

from sklearn.feature_extraction.text import CountVectorizer

import csv

CLASSES = [
    'treatment',  # 0
    'diagnosis',  # 1
    'prevention',  # 2
    'mechanism',  # 3
    'transmission',  # 4
    'epidemic forecasting',  # 5
    'case report'  # 6
]

TITLE_INDEX = 2
ABSTRACT_INDEX = 3


def int_to_class(val: int) -> list:
    ans = []

    for i in range(len(CLASSES)):
        if ((val >> i) & 1) == 1:
            ans.append(CLASSES[i])

    return ans


def update_labels(original_data):
    for i in range(len(original_data)):

        labels = original_data[i][-1]
        labels = labels.lower()
        labels = labels.split(';')

        original_data[i][-1] = 0

        for j in range(len(CLASSES)):
            if CLASSES[j] in labels:
                original_data[i][-1] = original_data[i][-1] | (1 << j)

        assert original_data[i][-1]


def get_train_and_test_data():
    train_filename = "./Datasets/BC7-LitCovid-Train.csv"
    test_filename = "./Datasets/BC7-LitCovid-Dev.csv"

    train_data = csv.reader(open(train_filename, "rt"))
    train_data = list(train_data)
    train_data = train_data[1:]

    test_data = csv.reader(open(test_filename, "rt"))
    test_data = list(test_data)
    test_data = test_data[1:]

    update_labels(train_data)
    update_labels(test_data)

    train_data = [[row[TITLE_INDEX] + ' ' + row[ABSTRACT_INDEX], row[-1]] for row in train_data]
    test_data = [[row[TITLE_INDEX] + ' ' + row[ABSTRACT_INDEX], row[-1]] for row in test_data]

    return train_data, test_data


trainingData, testData = get_train_and_test_data()

# pp.pprint(training_data)
# pp.pprint(test_data)

print(np.shape(trainingData))

pp.pprint(trainingData[0][1])

from nltk.tokenize import word_tokenize

new_sentence = 'what is the price of the book'
new_word_list = word_tokenize(new_sentence)

columns = ['sent', 'class']
rows = []

rows = trainingData
training_data = pd.DataFrame(trainingData[0], columns=columns)
# pp.pprint(training_data)

class_names = training_data['class'].unique()

print("CLASS NAMES")
pp.pprint(class_names)

classFreqs = training_data.groupby('class').count() / training_data.shape[0]
print("CLASSFREQS")
pp.pprint(classFreqs)

pp.pprint(training_data['class'])

docs = [row['sent'] for index, row in training_data.iterrows()]

vec = CountVectorizer()
X = vec.fit_transform(docs)

total_features = len(vec.get_feature_names())
# pp.pprint(total_features)


tdms = []
freqs = []
probs = []
answers = []
newWords = []
for idx, label in enumerate(class_names):

    class_docs = [row['sent'] for index, row in training_data.iterrows() if row['class'] == label]

    vec_s = CountVectorizer()
    X_s = vec_s.fit_transform(class_docs)
    tdms.append(pd.DataFrame(X_s.toarray(), columns=vec_s.get_feature_names()))

    word_list_s = vec_s.get_feature_names();
    count_list_s = X_s.toarray().sum(axis=0)

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
        # print(word,newWords[-1][word])
    answers.append(answer * classFreqs['sent'][idx + 1])

# pp.pprint(tdm_s)

pp.pprint(tdms)
pp.pprint(freqs)
pp.pprint(probs)
pp.pprint(answers)
pp.pprint(newWords)

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
