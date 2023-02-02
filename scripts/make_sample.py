import sys

import numpy as np
import pandas as pd

def main():
    # set seed
    np.random.seed(seed=32)

    # make data
    data_num = 5
    dim = 18
    mu = np.random.randint(-5, 5, (3, dim))
    sample_ary0 = np.full((data_num, dim+1), 0.0)
    sample_ary0[:, 1:] = np.random.randn(data_num, dim) + mu[0, :]
    sample_ary1 = np.full((data_num, dim+1), 1.0)
    sample_ary1[:, 1:] = np.random.randn(data_num, dim) + mu[1, :]
    sample_ary2 = np.full((data_num, dim+1), 2.0)
    sample_ary2[:, 1:] = np.random.randn(data_num, dim) + mu[2, :]
    sample_ary = np.concatenate([sample_ary0, sample_ary1, sample_ary2])

    df = pd.DataFrame(sample_ary)

    df.to_csv('data/sample_test.csv')

if __name__ == "__main__":
    main()
