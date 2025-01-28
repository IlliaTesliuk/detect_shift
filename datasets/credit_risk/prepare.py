import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

ROOT_DIR = "../.."


def prepare():
    data = pd.read_csv(f"{ROOT_DIR}/datasets/credit_risk/credit_risk_dataset.csv")

    data.drop_duplicates(inplace=True)

    data.dropna(axis=0, inplace=True)

    data.reset_index(inplace=True)

    data = data.drop(data[data["person_age"] > 80].index, axis=0)

    data = data.drop(data[data["person_emp_length"] > 60].index, axis=0)
    data = data.drop(["index"], axis=1)
    data.reset_index(inplace=True)
    data = data.drop(["index"], axis=1)

    data["loan_to_income_ratio"] = data["loan_amnt"] / data["person_income"]
    data["loan_to_emp_length_ratio"] = data["person_emp_length"] / data["loan_amnt"]
    data["int_rate_to_loan_amt_ratio"] = data["loan_int_rate"] / data["loan_amnt"]

    X = data.drop(["loan_status"], axis=1)
    Y = data["loan_status"].array

    X.reset_index(inplace=True)
    X = X.drop(["index"], axis=1)

    ohe_colums = [
        "cb_person_default_on_file",
        "loan_grade",
        "person_home_ownership",
        "loan_intent",
    ]
    ohe = OneHotEncoder()
    ohe.fit(X[ohe_colums])

    merge_ohe_col = [x.lower() for x in np.concatenate([x for x in ohe.categories_])]
    ohe_data = pd.DataFrame(
        ohe.transform(X[ohe_colums]).toarray(), columns=merge_ohe_col
    )
    X_new = pd.concat([ohe_data, X], axis=1)
    X_new = X_new.drop(ohe_colums, axis=1)

    normal_col = [
        "person_income",
        "person_age",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "cb_person_cred_hist_length",
        "loan_percent_income",
        "loan_to_emp_length_ratio",
        "int_rate_to_loan_amt_ratio",
    ]

    scaler_normal = StandardScaler()
    X_new.loc[:, normal_col] = scaler_normal.fit_transform(X_new.loc[:, normal_col])

    data = np.hstack([X_new, Y.reshape(-1, 1)])

    with open(f"{ROOT_DIR}/datasets/credit_risk/train_clean.npy", "wb") as fout:
        np.save(fout, data)


if __name__ == "__main__":
    prepare()
