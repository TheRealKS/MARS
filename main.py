import string
import sys

import numpy as np
import pandas as pd

from MARS import runMARSForward, runMARSBackward

import logging
import pickle

SSR = 222.45642923028186


def main():
    logging.basicConfig(filename='log.txt', filemode='w', level=logging.WARNING)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    df = pd.read_json('dataset.json')
    X = df['X'].to_numpy()
    y = df['y'].to_numpy()
    X = np.array(list(map(np.array, X)))
    X = X[:1000]
    y = y[:1000]
    labels = gen_labels(len(X[0]))
    model, ssr = runMARSForward(X, y, labels, len(X[0]), maxSplits=10)
    with open("model.pickle", "wb") as outfile:
        # "wb" argument opens the file in binary mode
        print(model)
        pickle.dump(model.components, outfile)
        print(ssr)

def gen_labels(n):
    alphabet_string = string.ascii_lowercase
    alphabet_list = list(alphabet_string)
    return alphabet_list[:n]

if __name__ == '__main__':
    main()
