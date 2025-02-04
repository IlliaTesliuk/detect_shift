import numpy as np
import pandas as pd
from scipy.linalg import logm
from numpy.linalg import eig
from copy import deepcopy
from sklearn.model_selection import train_test_split


def cdd_test(X_src, y_src, X_tgt, y_tgt, n_boot, seed):

    Xy_src = np.concatenate((X_src, y_src.reshape((-1, 1))), 1)
    Xy_tgt = np.concatenate((X_tgt, y_tgt.reshape((-1, 1))), 1)

    p = X_tgt.shape[1]

    # Cxy_src = corrent_matrix(Xy_src, kernel_size=0.05)
    # Cxy_tgt = corrent_matrix(Xy_tgt, kernel_size=0.05)
    # Cx_src = Cxy_src[0:p, 0:p]
    # Cx_tgt = Cxy_tgt[0:p, 0:p]

    # Cxy_s = pairwise_kernels(np.transpose(Xys), metric='rbf',gamma=None)
    # Cxy_t = pairwise_kernels(np.transpose(Xyt), metric='rbf',gamma=None)
    # Cx_s = Cxy_s[0:p,0:p]
    # Cx_t = Cxy_t[0:p,0:p]

    Cxy_src = np.cov(Xy_src, rowvar=False)
    Cxy_tgt = np.cov(Xy_tgt, rowvar=False)
    Cx_src = Cxy_src[0:p, 0:p]
    Cx_tgt = Cxy_tgt[0:p, 0:p]

    cdd = 0.5 * (
        log_det_divergenceEigSort(Cxy_src, Cxy_tgt)
        - log_det_divergenceEigSort(Cx_src, Cx_tgt)
        + log_det_divergenceEigSort(Cxy_tgt, Cxy_src)
        - log_det_divergenceEigSort(Cx_tgt, Cx_src)
    )

    X_all = np.concatenate((X_src, X_tgt), 0)
    y_all = np.concatenate((y_src, y_tgt), 0)

    cdd_null = np.zeros(n_boot)

    for b in np.arange(n_boot):

        X1, X2, y1, y2 = train_test_split(X_all, y_all, test_size=0.5, random_state=b)

        Xy1 = np.concatenate((X1, y1.reshape((-1, 1))), 1)
        Xy2 = np.concatenate((X2, y2.reshape((-1, 1))), 1)
        p = X1.shape[1]
        # Cxy_1 = pairwise_kernels(np.transpose(Xy1), metric='rbf',gamma=None)
        # Cxy_2 = pairwise_kernels(np.transpose(Xy2), metric='rbf',gamma=None)
        Cxy_1 = np.cov(Xy1, rowvar=False)
        Cxy_2 = np.cov(Xy2, rowvar=False)
        Cx_1 = Cxy_1[0:p, 0:p]
        Cx_2 = Cxy_2[0:p, 0:p]
        v1 = log_det_divergenceEigSort(Cxy_1, Cxy_2)
        v2 = log_det_divergenceEigSort(Cx_1, Cx_2)
        v3 = log_det_divergenceEigSort(Cxy_2, Cxy_1)
        v4 = log_det_divergenceEigSort(Cx_2, Cx_1)
        # print(v1, v2, v3, v4)
        # print(type(v1), type(v2), type(v3), type(v4))
        ss = v1 - v2 + v3 - v4
        cdd_null[b] = 0.5 * ss.real
        # cdd_null[b] = 0.5 * (
        #    log_det_divergenceEigSort(Cxy_1, Cxy_2)
        #    - log_det_divergenceEigSort(Cx_1, Cx_2)
        #    + log_det_divergenceEigSort(Cxy_2, Cxy_1)
        #    - log_det_divergenceEigSort(Cx_2, Cx_1)
        # )

    pv = (1 + len(np.where(cdd_null > cdd)[0])) / (1 + n_boot)

    return pv


################### Kernel functions


def RBFkernel(x1, x2, alpha=0.5):
    """Computing the RBF kernel between two vectors x1 and x2.
    K(x1 ,x2 )=\exp (-\frac {\|x1-x2\|^2}{2alpha^2})

    ----------
    x1 : np.array
        the first sample
    x2 : np.array
        the second sample
    alpha: float
        the kernel width

    Returns
    -------
     K(x1,x2) : float
        the RBF kernel result
    """

    return np.exp(-np.sum(np.power(x1 - x2, 2)) / (2 * alpha**2))


def linearkernel(x1, x2):
    """Computing the linear kernel between two vectors x1 and x2.
    K(x1 ,x2 )  = x1^T x2
    ----------
    x1 : np.array
        the first sample
    x2 : np.array
        the second sample

    Returns
    -------
     K(x1,x2) : float
        the linear kernel result
    """
    return np.sum(np.dot(x1, x2))


################### Useful matrix functions
def is_pos_def(x):
    """Checks if the matrix x is positive semidefinite by checking that all eigenvalues are >=0
    ----------
    x : np.array
        the matrix to be checked

    Returns
    -------
      : Boolean
        whether the matrix x is positive semidefinite or not
    """
    return np.all(np.linalg.eigvals(x) >= 0)


def min_eigvals(x):
    """Returns the minimum eigenvalues of matrix x
    ----------
    x : np.array
        the matrix to be checked

    Returns
    -------
      : float
        the smallest eigenvalue of matrix x
    """
    return min(np.linalg.eigvals(x))


def sample_Cov_Mat(dim):
    """Returns a random positive semidefinite covariate matrix
    ----------
    dim : int
        the dimension of the required matrix

    Returns
    -------
    C : np.array
        a random positive semidefinite covariate matrix
    """
    a = np.random.uniform(-1, 1, size=(dim, dim))
    O, r = np.linalg.qr(a, mode="complete")
    p = np.random.uniform(-1, 1, dim)
    p = np.sort(p)
    D = np.diag(np.square(p))
    C = np.matmul(np.matmul(O.transpose(), D), O)
    return C


################### von Neumann divergence functions


def von_Neumann_divergence(A, B):
    """Computing the von Neumann divergence between two positive semidefinite matrices A and B
    D_{vN}(A||B) = Tr(A (log(A)-log(B))-A+B)
    ----------
    A : np.array
        the first array
    B : np.array
        the second array

    Returns
    -------
      : float
        the von Neumann divergence
    """
    return np.trace(A.dot(logm(A) - logm(B)) - A + B)


def von_Neumann_divergence_Eff(A, B):
    """Computing the von Neumann divergence between two positive semidefinite matrices A and B efficiently
    D_{vN}(A||B) = Tr(A (log(A)-log(B))-A+B)
    ----------
    A : np.array
        the first array
    B : np.array
        the second array

    Returns
    -------
      : float
        the von Neumann divergence
    """
    # Divergence = np.trace(np.dot(A, logm(A)) - np.dot(A, logm(B)) - A + B)
    Aeig_val, Aeig_vec = eig(A)
    Beig_val, Beig_vec = eig(B)
    Aeig_val, Aeig_vec = abs(Aeig_val), (Aeig_vec)
    Beig_val, Beig_vec = abs(Beig_val), (Beig_vec)
    Aeig_val[Aeig_val < 1e-10] = 0
    Beig_val[Beig_val < 1e-10] = 0

    A_val_temp, B_val_temp = deepcopy(Aeig_val), deepcopy(Beig_val)
    A_val_temp[Aeig_val <= 0] = 1
    B_val_temp[Beig_val <= 0] = 1

    part1 = np.sum(Aeig_val * np.log2(A_val_temp) - Aeig_val + Beig_val)

    lambda_log_theta = np.dot(
        Aeig_val.reshape(len(Aeig_val), 1),
        np.log2(B_val_temp.reshape(1, len(B_val_temp))),
    )
    part2 = (np.dot(Aeig_vec.T, Beig_vec) ** 2) * lambda_log_theta
    part2 = -np.sum(part2)
    Divergence = part1 + part2

    return Divergence


################### log det divergence
def log_det_divergence(A, B):
    """Computing the logDet divergence between two positive semidefinite matrices A and B
    D_{\ell D}(A||B) = \Tr(B^{-1}A) + \log_2\frac{|B|}{|A|} - n,
    ----------
    A : np.array
        the first array
    B : np.array
        the second array

    Returns
    -------
      : float
        the logDet divergence
    """

    cross_term = (
        np.trace(np.matmul(A, np.linalg.inv(B)))
        - np.log(np.linalg.det(np.matmul(A, np.linalg.inv(B))))
        - A.shape[0]
    )
    return cross_term


def log_det_divergenceEigSort(A, B):
    """Computing the logDet divergence between two positive semidefinite matrices A and B efficiently
    D_{\ell D}(A||B) = \Tr(B^{-1}A) + \log_2\frac{|B|}{|A|} - n,
    ----------
    A : np.array
        the first array
    B : np.array
        the second array

    Returns
    -------
      : float
        the logDet divergence
    """
    # print(np.linalg.cond(A), np.linalg.det(A), np.linalg.cond(B), np.linalg.det(B))
    Aeig_val, Aeig_vec = eig(A)
    # print(np.iscomplexobj(Aeig_vec), np.iscomplexobj(Aeig_val))
    idx = Aeig_val.argsort()[::-1]
    Aeig_val = Aeig_val[idx]
    Aeig_vec = Aeig_vec[:, idx]

    Beig_val, Beig_vec = eig(B)
    idx = Beig_val.argsort()[::-1]
    Beig_val = Beig_val[idx]
    Beig_vec = Beig_vec[:, idx]

    Aeig_val = abs(Aeig_val)
    Beig_val = abs(Beig_val)
    Aeig_val[Aeig_val < 1e-10] = 0
    Beig_val[Beig_val < 1e-10] = 0
    length = A.shape[0]
    cross_term = 0
    for i in range(length):
        for j in range(length):
            cross_term += (Aeig_vec[:, i].dot(Beig_vec[:, j]) ** 2) * (
                Aeig_val[i] / Beig_val[j]
                if (Aeig_val[i] > 0 and Beig_val[j] > 0)
                else 1
            )
        cross_term -= (
            np.log(Aeig_val[i] / Beig_val[i])
            if (Aeig_val[i] > 0 and Beig_val[i] > 0)
            else 1
        )

    return cross_term - length


################### Centered Correntropy divergence


def corrent_matrix(data, kernel_size):
    """
    data: np.array
        data of size n x d, n is number of sample, d is dimension
    kernel_size: float
        the kernel width
    -------
    data: np.array
        a d x d (symmetric) center correntropy matrix
    """
    dim = data.shape[1]
    corren_matrix = np.zeros(shape=(dim, dim))
    for i in range(dim):
        for j in range(i + 1):
            corren_matrix[i, j] = corren_matrix[j, i] = sample_center_correntropy(
                data[:, i], data[:, j], kernel_size
            )

    return corren_matrix


def sample_center_correntropy(x, y, kernel_size):
    """Computing the center correntropy between two vectors x and y
    ----------
    x : np.array
        the first sample
    y : np.array
        the second sample
    kernel_size: float
        the kernel width

    Returns
    -------
      : float
        center correntropy between X and Y
    """

    twosquaredSize = 2 * kernel_size**2
    bias = 0
    for i in range(x.shape[0]):
        bias += sum(np.exp(-((x[i] - y) ** 2) / twosquaredSize))
        # for j in range(x.shape[0]):
        #    bias +=np.exp(-(x[i]-y[j])**2/twosquaredSize)
    bias = bias / x.shape[0] ** 2

    corren = (1 / x.shape[0]) * sum(np.exp(-((x - y) ** 2) / twosquaredSize)) - bias
    return corren
