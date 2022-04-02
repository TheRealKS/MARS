import sys

import numpy as np
import pandas as pd

from MARS import runMARSForward, runMARSBackward

import logging
import pickle

from MARSModel import MARSModel

SSR = 5331.59876407358


def main():
    logging.basicConfig(filename='log.txt', filemode='w', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    df = pd.read_json('dataset.json')
    X = df['X'].to_numpy()
    y = df['y'].to_numpy()
    X = np.array(list(map(np.array, X)))
    labels = ['a', 'b', 'c', 'd', 'f']

    with open("model.pickle", "rb") as infile:
        model = pickle.load(infile)
        m = MARSModel()
        m.components = model
        old = m.copy()
        newmodel = runMARSBackward(m, SSR, X, y, len(X[0]), maxSplits=10)
        print(old)
        print(newmodel)


if __name__ == '__main__':
    main()
