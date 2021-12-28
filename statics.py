import re
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

def get_vocab_size(data) -> int:
    longString = ""
    for text in data:
        longString += text[0].lower() + " "
    
    with open('stopwords.txt') as fp: # Load Stopwords
        stopwords = set(fp.read().splitlines())

    vocab = set(re.findall(r"[a-zA-Z]+", longString))
    vocab = vocab.difference(stopwords)

    return len(vocab)


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

def get_number_of_docs_in_class(data, index:int) -> int:
    ans = 0
    
    for i in range(len(data)):
        if ((data[i][-1] >> index) & 1) == 1:
            ans += 1
    
    return ans

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

def get_statics(train_data, test_data):
    print("FOR TRAIN DATA")
    print("Number of Documents:", len(train_data))
    print("Number of distinct words", get_vocab_size(train_data))
    for i in range(len(CLASSES)):
        print("{} documents have class {}".format(get_number_of_docs_in_class(train_data, i), CLASSES[i]))
    print("FOR TEST DATA")
    print("Number of Documents:", len(test_data))
    print("Number of distinct words", get_vocab_size(test_data))
    for i in range(len(CLASSES)):
        print("{} documents have class {}".format(get_number_of_docs_in_class(test_data, i), CLASSES[i]))
    

train_data, test_data = get_train_and_test_data()
get_statics(train_data, test_data)
