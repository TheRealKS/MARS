import sys

import numpy as np
import pandas as pd

from MARS import runMARSForward, runMARSBackward, evalModelWithCoefs

from MARSModel import *

import logging
import string
import pickle

from sklearn.model_selection import train_test_split

SSR = 5331.59876407358


def main():
    logging.basicConfig(filename='log.txt', filemode='w', level=logging.WARNING)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    weather_df = pd.read_csv('weatherHistory.csv')
    weather_df.loc[weather_df['Precip Type'] == 'rain', 'Precip Type'] = 1
    weather_df.loc[weather_df['Precip Type'] == 'snow', 'Precip Type'] = 0
    weather_df_num = weather_df[list(weather_df.dtypes[weather_df.dtypes != 'object'].index)]
    weather_y = weather_df_num.pop('Temperature (C)')
    weather_X = weather_df_num
    train_X, test_X, train_y, test_y = train_test_split(weather_X, weather_y, test_size=0.2, random_state=4)
    train_X = weather_X[:100]['Apparent Temperature (C)'].to_numpy()
    train_X = train_X[..., None]
    train_y = weather_X[:100]['Humidity'].to_numpy()
    test_X = test_X.to_numpy()
    test_y = test_y.to_numpy()
    labels = ['a']
    maxSplits = 3
    model, ssr = runMARSForward(train_X, train_y, labels, maxSplits=maxSplits)
    with open("model_trunc_" + str(maxSplits) + ".pickle", "wb") as outfile:
        # "wb" argument opens the file in binary mode
        print(model)
        d = {
            'model': model,
            'ssr': ssr
        }
        pickle.dump(d, outfile)
        print(ssr)

    newmodel, coefs = runMARSBackward(model, ssr, train_X, labels, train_y, len(train_X), maxSplits=maxSplits)
    print(getModelLength(newmodel))
    with open("model_aug.pickle", "wb") as outfile:
        # "wb" argument opens the file in binary mode
        print(model)
        d = {
            'model': newmodel,
            'c': coefs
        }
        pickle.dump(d, outfile)
        print(coefs)

def gen_labels(n):
    alphabet_string = string.ascii_lowercase
    alphabet_list = list(alphabet_string)
    return alphabet_list[:n]

if __name__ == '__main__':
    main()
