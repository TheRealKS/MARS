import sys

import numpy as np
import pandas as pd

from MARS import runMARSForward

import logging


def main():
    logging.basicConfig(filename='log.txt', filemode='w', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    df = pd.read_json('dataset.json')
    print(df)
    X = df['X'].to_numpy()
    y = df['y'].to_numpy()
    X = np.array(list(map(np.array, X)))
    print(X)
    print(y)
    labels = ['a', 'b', 'c', 'd', 'f']
    model = runMARSForward(X, y, labels, 5, maxSplits=8)
    print(model)


if __name__ == '__main__':
    main()
