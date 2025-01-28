import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT_DIR = "../.."


def prepare():
    data = pd.read_csv(f"{ROOT_DIR}/datasets/credit_card_default/UCI_Credit_Card.csv")

    data.rename(columns={"default.payment.next.month": "Default"}, inplace=True)
    # drop column "ID"
    data.drop("ID", axis=1, inplace=True)

    y = data.Default  # target default=1 or non-default=0

    # The categories 4:others, 5:unknown, and 6:unknown can be grouped into a single class '4'.
    data["EDUCATION"] = np.where(data["EDUCATION"] == 5, 4, data["EDUCATION"])
    data["EDUCATION"] = np.where(data["EDUCATION"] == 6, 4, data["EDUCATION"])
    data["EDUCATION"] = np.where(data["EDUCATION"] == 0, 4, data["EDUCATION"])

    # category '0' which will be joined to the category '3' = others.
    data["MARRIAGE"] = np.where(data["MARRIAGE"] == 0, 3, data["MARRIAGE"])

    X = data.drop("Default", axis=1)
    y = data["Default"].to_numpy()

    # scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    data = np.hstack([X, y.reshape(-1, 1)])
    with open(f"{ROOT_DIR}/datasets/credit_card_default/train_clean.npy", "wb") as fout:
        np.save(fout, data)


if __name__ == "__main__":
    prepare()
