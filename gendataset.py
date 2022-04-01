from sklearn import datasets
import pandas as pd
import numpy as np

def main():
    X, y = datasets.make_friedman1(n_features=5)
    data = {'X': np.ndarray.tolist(X), 'y': y}
    df = pd.DataFrame(data)
    df.to_json('dataset.json')


if __name__ == '__main__':
    main()
