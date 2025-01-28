import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

ROOT_DIR = "../.."


def main():
    data = pd.read_csv(f"{ROOT_DIR}/datasets/adult_income/adult.csv")

    data["education"].replace("Preschool", "dropout", inplace=True)
    data["education"].replace("10th", "dropout", inplace=True)
    data["education"].replace("11th", "dropout", inplace=True)
    data["education"].replace("12th", "dropout", inplace=True)
    data["education"].replace("1st-4th", "dropout", inplace=True)
    data["education"].replace("5th-6th", "dropout", inplace=True)
    data["education"].replace("7th-8th", "dropout", inplace=True)
    data["education"].replace("9th", "dropout", inplace=True)
    data["education"].replace("HS-Grad", "HighGrad", inplace=True)
    data["education"].replace("HS-grad", "HighGrad", inplace=True)
    data["education"].replace("Some-college", "CommunityCollege", inplace=True)
    data["education"].replace("Assoc-acdm", "CommunityCollege", inplace=True)
    data["education"].replace("Assoc-voc", "CommunityCollege", inplace=True)
    data["education"].replace("Bachelors", "Bachelors", inplace=True)
    data["education"].replace("Masters", "Masters", inplace=True)
    data["education"].replace("Prof-school", "Masters", inplace=True)
    data["education"].replace("Doctorate", "Doctorate", inplace=True)

    # Limit categorization
    data["marital-status"].replace("Never-married", "NotMarried", inplace=True)
    data["marital-status"].replace(["Married-AF-spouse"], "Married", inplace=True)
    data["marital-status"].replace(["Married-civ-spouse"], "Married", inplace=True)
    data["marital-status"].replace(
        ["Married-spouse-absent"], "NotMarried", inplace=True
    )
    data["marital-status"].replace(["Separated"], "Separated", inplace=True)
    data["marital-status"].replace(["Divorced"], "Separated", inplace=True)
    data["marital-status"].replace(["Widowed"], "Widowed", inplace=True)

    # remove duplicated row
    data = data.drop_duplicates()
    # replace ? to nan
    data.replace("?", np.nan, inplace=True)

    df = data

    # drop beacuse they have nan
    df["occupation"].dropna(inplace=True)
    df["workclass"].dropna(inplace=True)

    # drop educational-num beacuse its not important
    df = df.drop(["educational-num"], axis=1)

    # Encoder cetegorical columns
    lb = LabelEncoder()
    df.workclass = lb.fit_transform(df.workclass)
    df.education = lb.fit_transform(df.education)
    df["marital-status"] = lb.fit_transform(df["marital-status"])
    df.occupation = lb.fit_transform(df.occupation)
    df.relationship = lb.fit_transform(df.relationship)
    df.race = lb.fit_transform(df.race)
    df.gender = lb.fit_transform(df.gender)
    df["native-country"] = lb.fit_transform(df["native-country"])
    df.income = lb.fit_transform(df.income)

    df = df.drop(["fnlwgt", "native-country"], axis=1)

    X = df.drop("income", axis=1)
    y = df["income"].to_numpy()

    # scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    data = np.hstack([X, y.reshape(-1, 1)])
    with open(f"{ROOT_DIR}/datasets/adult_income/train_clean.npy", "wb") as fout:
        np.save(fout, data)


if __name__ == "__main__":
    main()
