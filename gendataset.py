from sklearn import datasets
import pandas as pd
import numpy as np

# Generate a simple synthetic dataset. We use the sklearn make_friedman1 function for this, because it generates
# datasets using a method described in the MARS paper

def main():
    X = np.arange(10, 21, 0.2)
    y = [-37 + 5.1 * x for x in X]
    X, y = datasets.make_friedman1(n_features=5, n_samples=2500)
    data = {'X': np.ndarray.tolist(X), 'y': y}
    df = pd.DataFrame(data)
    df.to_json('dataset.json')


if __name__ == '__main__':
    main()
