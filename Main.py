import sys

from FileReader import write_dump_files, read_dump_files, Document
from MNB_KNN import apply_mnb, apply_knn

IS_LABEL_INT = False

nb = 1
knn = 1
only_read = 0


def main():
    global nb
    global knn
    global only_read

    if len(sys.argv) == 1:
        nb = 1
        knn = 1

    if len(sys.argv) > 1:
        for i in range(len(sys.argv)):
            if i == 0:
                continue
            arg = sys.argv[i]
            if arg == "nb":
                knn = 0
                nb = 1
            elif arg == "r":
                only_read = 1
            elif arg == "knn":
                nb = 0
                knn = 1

    if only_read == 0:
        write_dump_files()

    train_data, test_data = read_dump_files()

    if nb == 1:
        apply_mnb(train_data, test_data, IS_LABEL_INT)

    if knn == 1:
        apply_knn(train_data, test_data, [9], False)


main()
