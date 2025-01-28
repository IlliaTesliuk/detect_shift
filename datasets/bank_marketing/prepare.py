import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OneHotEncoder,
    MinMaxScaler,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from scipy import sparse


ROOT_DIR = "../.."


def get_encoder_inst(feature_col):

    assert isinstance(feature_col, pd.Series)
    feature_vec = feature_col.sort_values().values.reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(feature_vec)
    return enc


def get_one_hot_enc(feature_col, enc, cols):

    assert isinstance(feature_col, pd.Series)
    assert isinstance(enc, OneHotEncoder)
    unseen_vec = feature_col.values.reshape(-1, 1)
    encoded_vec = enc.transform(unseen_vec).toarray()
    column_name = enc.get_feature_names_out([cols])
    encoded_df = pd.DataFrame(encoded_vec, columns=column_name)
    return encoded_df


def prepare():
    df = pd.read_csv(f"{ROOT_DIR}/datasets/bank_marketing/bank.csv")

    # Label Encoder
    le = LabelEncoder()
    df.marital = le.fit_transform(df.marital)
    df.housing = le.fit_transform(df.housing)
    df.deposit = le.fit_transform(df.deposit)
    df.loan = le.fit_transform(df.loan)
    df.default = le.fit_transform(df.default)

    # One-Hot Encoder
    ohe_cat_list = ["job", "education", "month", "contact", "poutcome"]
    ohe_cat_data = df[ohe_cat_list]
    df.drop(ohe_cat_list, axis=1, inplace=True)

    data_list = []
    for cols in ohe_cat_data.columns:
        encoder = get_encoder_inst(ohe_cat_data[cols])
        one = get_one_hot_enc(ohe_cat_data[cols], encoder, cols)
        data_list.append(one)

    final_ohe = pd.concat(data_list, axis=1)
    df.reset_index(drop=True, inplace=True)
    final_ohe.reset_index(drop=True, inplace=True)
    for cols in final_ohe.columns:
        final_ohe[cols] = final_ohe[cols].astype("int")

    df = pd.concat([df, final_ohe], axis=1)

    X = df.drop("deposit", axis=1)
    y = df[["deposit"]]

    # scale features
    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)

    data = np.hstack([scaled_X, y])
    with open(f"{ROOT_DIR}/datasets/bank_marketing/train_clean.npy", "wb") as fout:
        np.save(fout, data)


def main():
    prepare()
    return
    df = pd.read_csv(f"{ROOT_DIR}/datasets/bank_marketing/bank.csv")
    term_deposits = df.copy()

    dep = term_deposits["deposit"]
    term_deposits.drop(labels=["deposit"], axis=1, inplace=True)
    term_deposits.insert(0, "deposit", dep)

    # Making pipelines
    numerical_pipeline = Pipeline(
        [
            (
                "select_numeric",
                DataFrameSelector(
                    [
                        "age",
                        "balance",
                        "day",
                        "campaign",
                        "pdays",
                        "previous",
                        "duration",
                    ]
                ),
            ),
            ("std_scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "select_cat",
                DataFrameSelector(
                    [
                        "job",
                        "education",
                        "marital",
                        "default",
                        "housing",
                        "loan",
                        "contact",
                        "month",
                        "poutcome",
                    ]
                ),
            ),
            ("cat_encoder", CategoricalEncoder(encoding="onehot-dense")),
        ]
    )

    preprocess_pipeline = FeatureUnion(
        transformer_list=[
            ("numerical_pipeline", numerical_pipeline),
            ("categorical_pipeline", categorical_pipeline),
        ]
    )
    # print(term_deposits)

    data = preprocess_pipeline.fit_transform(term_deposits)
    print(type(data))
    print(data)  # .shape)


# A class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


# Definition of the CategoricalEncoder class, copied from PR #9151.
# Just run this cell, or copy it to your code, no need to try to
# understand every line.
# Code reference Hands on Machine Learning with Scikit Learn and Tensorflow by Aurelien Geron.
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        encoding="onehot",
        categories="auto",
        dtype=np.float64,
        handle_unknown="error",
    ):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        if self.encoding not in ["onehot", "onehot-dense", "ordinal"]:
            template = (
                "encoding should be either 'onehot', 'onehot-dense' "
                "or 'ordinal', got %s"
            )
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ["error", "ignore"]:
            template = "handle_unknown should be either 'error' or " "'ignore', got %s"
            raise ValueError(template % self.handle_unknown)

        if self.encoding == "ordinal" and self.handle_unknown == "ignore":
            raise ValueError(
                "handle_unknown='ignore' is not supported for" " encoding='ordinal'"
            )

        X = check_array(X, dtype=object, accept_sparse="csc", copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == "auto":
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == "error":
                        diff = np.unique(Xi[~valid_mask])
                        msg = (
                            "Found unknown categories {0} in column {1}"
                            " during fit".format(diff, i)
                        )
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        X = check_array(X, accept_sparse="csc", dtype=object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=int)
        X_mask = np.ones_like(X, dtype=bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == "error":
                    diff = np.unique(X[~valid_mask, i])
                    msg = (
                        "Found unknown categories {0} in column {1}"
                        " during transform".format(diff, i)
                    )
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == "ordinal":
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32), n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix(
            (data, (row_indices, column_indices)),
            shape=(n_samples, indices[-1]),
            dtype=self.dtype,
        ).tocsr()
        if self.encoding == "onehot-dense":
            return out.toarray()
        else:
            return out


if __name__ == "__main__":
    main()
