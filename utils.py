import os
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import mutual_info_classif
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Optional
import copy

"""
Counts instances of each class in the dataset
"""


def _class_len(dataset: torch.utils.data.Dataset):
    return dict(Counter(dataset.targets))


"""
Splits a torch.utils.data.Dataset dataset into train and test subsets while keeping the original class distribution.
"""


def _stratified_split(dataset, test_size, seed):
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )

    labels = np.array(dataset.targets)
    train_indices, test_indices = next(splitter.split(X=range(len(dataset)), y=labels))

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    train_subset.targets = labels[train_indices]
    train_subset.classes = dataset.classes

    test_subset = torch.utils.data.Subset(dataset, test_indices)
    test_subset.targets = labels[test_indices]
    test_subset.classes = dataset.classes

    return train_subset, test_subset


"""
Splits a torch.utils.data.Dataset dataset into two source and two target subsets while keeping the original class distribution.
"""


def source_target_split(data: torch.utils.data.Dataset, target_size: float, seed: int):
    source, target = _stratified_split(data, target_size, seed)
    return source, target
    # source1, source2 = _stratified_split(source, 0.5, seed)
    # target1, target2 = _stratified_split(target, 0.5, seed)
    # return source1, source2, target1, target2


def data_split(X, y, test_size: float, seed: int):
    len1 = int(len(y) * test_size)
    indices = np.arange(len(y))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:len1], indices[len1:]
    if type(X) == np.ndarray:
        X1, X2 = X[train_indices, :], X[test_indices, :]
    elif type(X) == pd.DataFrame:
        X1, X2 = X.iloc[train_indices, :], X.iloc[test_indices, :]

    if type(y) == np.ndarray:
        y1, y2 = y[train_indices], y[test_indices]
    elif type(y) == pd.Series:
        y1, y2 = y.iloc[train_indices], y.iloc[test_indices]

    return (X1, y1), (X2, y2)


def array_copy(dataset):  #: tuple[np.ndarray, np.ndarray]):
    return dataset[0].copy(), dataset[1].copy()


# def array_merge(
#    dataset1: tuple[np.ndarray, np.ndarray], dataset2: tuple[np.ndarray, np.ndarray]
# ):
def array_merge(dataset1, dataset2):
    X1, y1 = dataset1
    X2, y2 = dataset2
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    return X, y


# def array_mask(dataset: tuple[np.ndarray, np.ndarray], mask: np.ndarray):
def array_mask(dataset, mask):
    return dataset[0][mask], dataset[1][mask]


def subvert_label(y, s: float, seed: int):
    n_attacked = int(len(y) * s)
    indices = np.arange(len(y))

    np.random.seed(seed)
    indices_attacked = np.random.choice(indices, n_attacked, replace=False)

    if type(y) == np.ndarray:
        y[indices_attacked] = 1 - y[indices_attacked]
    elif type(y) == pd.Series:
        y.iloc[indices_attacked] = 1 - y.iloc[indices_attacked]

    # orig_labels = y[indices_attacked]
    # print(orig_labels)
    # new_labels = 1 - orig_labels
    # print(new_labels)
    # y[indices_attacked] = new_labels

    return y, indices_attacked


def dataset_random_split(
    dataset: torch.utils.data.Dataset, test_size: float, seed: int
):
    len1 = int(len(dataset) * test_size)
    # len2 = len(dataset) - len1
    # generator = torch.Generator().manual_seed(seed)
    # sub1, sub2 = torch.utils.data.random_split(dataset, [len1, len2], generator=generator)
    # return sub1, sub2

    labels = np.array(dataset.targets)
    indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:len1], indices[len1:]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    train_subset.targets = labels[train_indices]
    train_subset.classes = dataset.classes

    test_subset = torch.utils.data.Subset(dataset, test_indices)
    test_subset.targets = labels[test_indices]
    test_subset.classes = dataset.classes

    return train_subset, test_subset


def subset_indices(data, target, subset_size=1000, zero_class=0, one_class=1):
    if type(data) == pd.DataFrame:
        zero_ids = np.where(data[target] == zero_class)[0]
        one_ids = np.where(data[target] == one_class)[0]
    else:
        zero_ids = np.where(data == zero_class)[0]
        one_ids = np.where(data == one_class)[0]

    sel_zero_ids = np.random.choice(zero_ids, size=(subset_size,), replace=False)
    sel_one_ids = np.random.choice(one_ids, size=(subset_size,), replace=False)

    ids_2000 = np.concatenate([sel_zero_ids, sel_one_ids])
    np.random.shuffle(ids_2000)

    return ids_2000


def ds_to_binary(dataset: torch.utils.data.Dataset, classes_one, new_classes):
    """
    Converts a multiclass dataset into a binary given a list of original classes corresponding to '1'.
    """
    idx_one = np.array([dataset.class_to_idx[c] for c in classes_one])
    dataset.targets = torch.tensor(
        [1 if x.item() in idx_one else 0 for x in dataset.targets]
    )
    dataset.classes = new_classes
    return dataset


def bootstrap_sample(dataset, n_samples=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if len(dataset) == 1:
        n_samples = len(dataset) if n_samples is None else n_samples

        sampled_indices = np.random.choice(len(dataset), n_samples, replace=True)

        bootstrap = torch.utils.data.Subset(dataset, sampled_indices)
        bootstrap.targets = dataset.targets[sampled_indices]
        bootstrap.classes = dataset.classes
    else:
        X, y = dataset
        n_samples = len(y) if n_samples is None else n_samples

        sampled_indices = np.random.choice(len(y), n_samples, replace=True)
        if type(X) == np.ndarray:
            X_boot = X[sampled_indices, :]
        elif type(X) == pd.DataFrame:
            X_boot = X.iloc[sampled_indices, :]

        if type(y) == np.ndarray:
            y_boot = (y[sampled_indices],)
            y_boot = np.squeeze(y_boot)
        elif type(y) == pd.Series:
            y_boot = y.iloc[sampled_indices]

        bootstrap = (X_boot, y_boot)

    return bootstrap


"""
Samples elements of the torch.utils.data.Dataset (boostrap: with replacement) 
"""


def bootstrap_sample_torch(
    dataset: torch.utils.data.Dataset, n_samples=None, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    if n_samples is None:
        n_samples = len(dataset)

    sampled_indices = np.random.choice(len(dataset), n_samples, replace=True)

    bootstrap = torch.utils.data.Subset(dataset, sampled_indices)
    bootstrap.targets = dataset.targets[sampled_indices]
    bootstrap.classes = dataset.classes

    return bootstrap


def write_array(data: np.ndarray, fpath):
    with open(fpath, "wb") as fout:
        np.save(fout, data)


def load_array(fpath):
    with open(fpath, "rb") as fin:
        data = np.load(fpath)
    return data


def calculate_p_value(W_hat, W_hat_null, n_bootstrap=200):
    p_value = (1 + np.sum(W_hat_null > W_hat)) / (1 + n_bootstrap)
    return p_value


def load_w_hat(src_dir):
    data = load_array(f"{src_dir}/w_hat.npy")
    W_hat_null, W_hat = data[:-1], data[-1]
    return W_hat_null, W_hat


def KL_loss(P, Q, eps=1e-12):
    P = np.clip(P, eps, None)
    Q = np.clip(Q, eps, None)

    P = P / P.sum(axis=1, keepdims=True)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return np.mean(np.sum(P * np.log(P / Q), axis=1))


def compute_kl_divergence(q_s1_t2, q_t1_t2):
    # Small value to avoid log(0) or division by zero
    epsilon = 1e-10

    # Clip probabilities to avoid invalid values
    q_s1_t2 = np.clip(q_s1_t2, epsilon, 1 - epsilon)
    q_t1_t2 = np.clip(q_t1_t2, epsilon, 1 - epsilon)

    # Compute the KL divergence
    kl_divergence = q_s1_t2 * np.log(q_s1_t2 / q_t1_t2) + (1 - q_s1_t2) * np.log(
        (1 - q_s1_t2) / (1 - q_t1_t2)
    )

    # Return the mean of the KL divergence
    return np.mean(kl_divergence)


@dataclass
class ModelParams:
    name: Optional[str] = None
    device: Optional[str] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None


def get_model_params(params_dict_orig: dict):
    params_dict = copy.deepcopy(params_dict_orig)

    model_params = ModelParams(name=params_dict["name"])
    del params_dict["name"]

    if "device" in params_dict:
        device = params_dict["device"]
        del params_dict["device"]
    else:
        device = "cuda" if model_params.name in ["vgg16", "cnn"] else None
    model_params.device = device

    if "epochs" in params_dict:
        epochs = params_dict["epochs"]
        del params_dict["epochs"]
    else:
        epochs = 10
    model_params.epochs = epochs

    if "batch_size" in params_dict:
        batch_size = params_dict["batch_size"]
        del params_dict["batch_size"]
    else:
        batch_size = 32
    model_params.batch_size = batch_size

    return model_params, params_dict


def prepare_D_data(source1, source2, target1, target2, seed):
    src1 = array_copy(source1)
    n_src1 = len(src1[1])
    tgt = array_merge(array_copy(target1), array_copy(target2))

    X_D, y_D = array_merge(src1, tgt)
    Z = np.ones(len(y_D), dtype=y_D.dtype)
    Z[:n_src1] = 0
    D = np.hstack([X_D, y_D.reshape(-1, 1)])

    D_train, D_test = data_split(D, Z, test_size=0.5, seed=seed)
    one_mask = D_train[1] == 1
    zero_mask = D_train[1] == 0
    n_z1_train = np.sum(one_mask)
    n_z0_train = np.sum(zero_mask)

    one_mask_test = D_test[1] == 1
    n_z1_test = np.sum(one_mask_test)
    # print(n_z0_train, n_z1_train, "     ", n_z1_test)
    D_test1 = array_mask(D_test, D_test[1] == 1)

    return D_train, D_test1, n_z0_train, n_z1_train


def mi_filter(X, y, pmax=50):
    mi = np.zeros(X.shape[1])
    for j in np.arange(X.shape[1]):
        mi[j] = mutual_info_classif(X[:, j].reshape(-1, 1), y)
    sel = np.argsort(-mi)[0:pmax]
    return sel


def sigma(x):
    res = np.exp(x) / (1 + np.exp(x))
    return res
