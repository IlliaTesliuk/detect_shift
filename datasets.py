import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from torchvision.datasets import ImageFolder, MNIST
import torchvision.transforms as T
from utils import (
    ds_to_binary,
    source_target_split,
    data_split,
    array_copy,
    load_array,
    array_merge,
    array_mask,
    subvert_label,
    dataset_random_split,
    subset_indices,
    sigma,
)


class DatasetWrapper:
    def __init__(
        self, name, attack_split=0.0, src_split=0.5, sub_split=0.5, seed=None, **kwargs
    ):
        self.seed = seed
        self.dataset = self._load_data(name, **kwargs)
        self.dataset_name = name

        self.attack_split = attack_split
        self.src_split = src_split
        self.sub_split = sub_split
        self.seed = seed

        self.is_torch = len(self.dataset) == 1

        splits = self._prepare_data()
        self.source = splits[0]
        self.target = splits[1]
        self.source1 = splits[2]
        self.source2 = splits[3]
        self.target1 = splits[4]
        self.target2 = splits[5]
        self.ids_attacked = splits[6]

        data = self._prepare_m3_data()
        self.d_train = data[0]
        self.d_test1 = data[1]
        self.n_z0_train = data[2]
        self.n_z1_train = data[3]

    def _prepare_m3_data(self):
        if self.is_torch:
            return None, 0, 0, None, None

        src1 = array_copy(self.source1)
        n_src1 = len(src1[1])
        tgt = array_merge(array_copy(self.target1), array_copy(self.target2))

        X_D, y_D = array_merge(src1, tgt)
        Z = np.ones(len(y_D), dtype=y_D.dtype)
        Z[:n_src1] = 0
        D = np.hstack([X_D, y_D.reshape(-1, 1)])

        D_train, D_test = data_split(D, Z, test_size=0.5, seed=self.seed)
        one_mask = D_train[1] == 1
        zero_mask = D_train[1] == 0
        n_z1_train = np.sum(one_mask)
        n_z0_train = np.sum(zero_mask)

        one_mask_test = D_test[1] == 1
        n_z1_test = np.sum(one_mask_test)
        # print(n_z0_train, n_z1_train, "     ", n_z1_test)
        D_test1 = array_mask(D_test, D_test[1] == 1)

        return D_train, D_test1, n_z0_train, n_z1_train

    def _load_data(self, name, **kwargs):
        if name == "artificial":
            dataset = artificial(**kwargs)
        else:
            dataset = load_dataset(name)

        return dataset

    def _prepare_data(self):
        if self.is_torch:
            source, target = source_target_split(
                self.dataset, self.src_split, self.seed
            )

            y_orig = target.targets.copy()
            if self.attack_split > 0.0:
                n_attacked = int(len(target) * self.attack_split)

                np.random.seed(self.seed)
                ids_attacked = np.random.choice(
                    np.arange(n_attacked), n_attacked, replace=False
                )
                target.targets[ids_attacked] = 1 - y_orig[ids_attacked]

            source1, source2 = dataset_random_split(source, self.sub_split, self.seed)
            target1, target2 = dataset_random_split(target, self.sub_split, self.seed)

        else:
            X, y = self.dataset
            source, target = data_split(X, y, self.src_split, self.seed)
            X_src, y_src = source
            X_tgt, y_tgt = target
            y_orig = y_tgt.copy()

            ids_attacked = None
            if self.attack_split > 0.0:
                y_tgt, ids_attacked = subvert_label(y_tgt, self.attack_split, self.seed)

            source1, source2 = data_split(X_src, y_src, self.sub_split, self.seed)
            target1, target2 = data_split(X_tgt, y_tgt, self.sub_split, self.seed)

        return source, target, source1, source2, target1, target2, ids_attacked


def artificial(ns=2000, ncols=10, b=1, s=0, seed=None):
    mean1 = np.zeros(ncols)
    cov1 = np.zeros((ncols, ncols))
    for i in np.arange(ncols):
        for j in np.arange(ncols):
            cov1[i, j] = s ** (np.abs(i - j))

    np.random.seed(seed)
    X = np.random.multivariate_normal(mean=mean1, cov=cov1, size=ns)

    beta_true = np.repeat(b, ncols)
    eta = np.dot(X, beta_true)
    prob_true = sigma(eta)
    y = np.zeros(ns)

    for i in np.arange(0, ns, 1):
        y[i] = np.random.binomial(1, prob_true[i], size=1)

    return X, y


def load_dataset(ds_name):
    if ds_name == "credit_default":
        ds_name = "credit_card_default"

    if ds_name.endswith("binary"):
        ds_file = ds_name[: ds_name.find("_")] + "_train_embs_50.npy"
    else:
        ds_file = "train_clean.npy"
    ds_path = f"datasets/{ds_name}/{ds_file}"
    return _load_dataset(ds_path)


def _load_dataset(ds_path):
    data = load_array(ds_path)

    ids_2000 = subset_indices(data, None, subset_size=1000, zero_class=0, one_class=1)
    data = data[ids_2000, :]

    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y
