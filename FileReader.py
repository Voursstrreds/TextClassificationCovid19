import csv
import json
import string
import os

VOCABULARY = set()
IS_LABEL_INT = False

WORD_INDEX = {}
PROCESS_INDEXES = [
    2, # TITLE
    3, # ABSTRACT
]
PROCESS_FIELDS = {'title', 'abstract'}

CLASSES = [
    'treatment',  # 0
    'diagnosis',  # 1
    'prevention',  # 2
    'mechanism',  # 3
    'transmission',  # 4
    'epidemic forecasting',  # 5
    'case report'  # 6
]

TOTAL_CLASS_NUM = 7

with open('stopwords.txt') as fp: # Load Stopwords
    STOPWORDS = set(fp.read().splitlines())


class Document:
    def __init__(self, input :list):
        self.pmid: int = input[0]
        self.journal: str = input[1]
        self.title: list = input[2]
        self.abstract: list = input[3]
        self.keywords: list = input[4]
        self.pub_type: list = input[5]
        self.authors: list = input[6]
        self.doi: str = input[7]
        self.label: list = input[8]

    def get_bag_of_words(self):
        ret = {}
        words = self.title + self.abstract

        for word in words:
            if word in ret.keys():
                ret[word] = ret[word] + 1
            else:
                ret[word] = 1

        return ret

    def to_string(self):
        return_val = {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "label": self.label
        }
        return json.dumps(return_val)

    def is_equal(self, other):
        if self.pmid == other.pmid:
            return True
        return False


def get_train_and_test_data_as_list() -> tuple:
    train_filename = "./Datasets/BC7-LitCovid-Train.csv"
    test_filename = "./Datasets/BC7-LitCovid-Dev.csv"

    train_data = read_csv(train_filename)
    train_data = normalize_data(train_data, IS_LABEL_INT)

    test_data = read_csv(test_filename)
    test_data = normalize_data(test_data, IS_LABEL_INT)

    train_data = [Document(row) for row in train_data]
    test_data = [Document(row) for row in test_data]

    return train_data, test_data


def read_csv(file_name:str) -> list:
    data = csv.reader(open(file_name, "rt"))

    return list(data)[1:]


def normalize_data(data: list, labels_int: bool) -> list:
    for data_index in PROCESS_INDEXES:
        for i in range(len(data)):
            data[i][data_index] = get_words(data[i][data_index])

    update_labels(data, labels_int)

    return data


def get_words(input: str, is_remove_stopwords: bool = True) -> list:
    input_words = input.translate(str.maketrans('', '', string.punctuation)).lower()
    if is_remove_stopwords:
        input_words = remove_stopwords(input_words.split())

    return input_words


def remove_stopwords(words: list) -> list:
    ret = []

    for word in words:
        if word not in STOPWORDS:
            ret.append(word)

    return ret


def update_labels(original_data, is_int: bool):
    for i in range(len(original_data)):
        labels = original_data[i][-1]
        labels = labels.lower()
        labels = labels.split(';')

        original_data[i][-1] = encode_label(labels) if is_int else labels

        assert original_data[i][-1]


def encode_label(labels: str) -> int:
    ret = 0

    for j in range(len(CLASSES)):
        if CLASSES[j] in labels:
            ret = ret | (1 << j)

    return ret


def create_dump_path(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def dump(train_list: list, test_list: list):
    path = os.path.join(".", "dump_files", "")
    path_train = os.path.join(".", "dump_files", "train")
    path_test = os.path.join(".", "dump_files", "test")
    create_dump_path(path)

    docnum: int = 0
    train_dump = open(path_train, "w")
    to_write = {}
    for single_doc in train_list:
        single_dict = single_doc.to_string()
        to_write[docnum] = single_dict
        docnum = docnum + 1
    json.dump(to_write, train_dump)
    train_dump.close()

    docnum = 0
    test_dump = open(path_test, "w")
    to_write = {}
    for single_doc in test_list:
        single_dict = single_doc.to_string()
        to_write[docnum] = single_dict
        docnum = docnum + 1
    json.dump(to_write, test_dump)
    test_dump.close()


def write_dump_files():
    train_data, test_data = get_train_and_test_data_as_list()
    dump(train_data, test_data)


def read_dump_files():

    train_list = []
    test_list = []

    raw_file_train = open("./dump_files/train")
    raw = json.load(raw_file_train)
    for i in range(len(raw)):
        key = str(i)
        raw_doc = json.loads(raw[key])
        pmid = raw_doc["pmid"]
        title = raw_doc["title"]
        abstract = raw_doc["abstract"]
        label = raw_doc["label"]
        params = [pmid, "", title, abstract, "", "", "", "", label]
        train_list.append(Document(params))
    raw_file_train.close()

    raw_file_test = open("./dump_files/test")
    raw = json.load(raw_file_test)
    for i in range(len(raw)):
        key = str(i)
        raw_doc = json.loads(raw[key])
        pmid = raw_doc["pmid"]
        title = raw_doc["title"]
        abstract = raw_doc["abstract"]
        label = raw_doc["label"]
        params = [pmid, "", title, abstract, "", "", "", "", label]
        test_list.append(Document(params))
    raw_file_test.close()

    return train_list, test_list
