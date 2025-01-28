import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT_DIR = "../.."


def prepare():
    data = pd.read_csv(f"{ROOT_DIR}/datasets/pulsars/HTRU_2.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].to_numpy()

    # scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    data = np.hstack([X, y.reshape(-1, 1)])
    with open(f"{ROOT_DIR}/datasets/htru2/train_clean.npy", "wb") as fout:
        np.save(fout, data)


if __name__ == "__main__":
    prepare()
