import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import train_test_split


def mmd_test(X_src, y_src, X_tgt, y_tgt, n_boot, seed, gamma=None):

    Xy_src = np.concatenate((X_src, y_src.reshape((-1, 1))), 1)
    Xy_tgt = np.concatenate((X_tgt, y_tgt.reshape((-1, 1))), 1)
    ns = Xy_src.shape[0]
    nt = Xy_tgt.shape[0]

    term_s = np.sum(pairwise_kernels(Xy_src, metric="rbf", gamma=gamma))
    term_t = np.sum(pairwise_kernels(Xy_tgt, metric="rbf", gamma=gamma))
    term_mix = np.sum(pairwise_kernels(Xy_src, Xy_tgt, metric="rbf", gamma=gamma))

    mmd = (
        (1 / (ns**2)) * term_s
        + (1 / (nt**2)) * term_t
        - 2 * (1 / (ns)) * (1 / (nt)) * term_mix
    )

    X_all = np.concatenate((X_src, X_tgt), 0)
    y_all = np.concatenate((y_src, y_tgt), 0)

    mmd_null = np.zeros(n_boot)

    for b in np.arange(n_boot):

        X1, X2, y1, y2 = train_test_split(X_all, y_all, test_size=0.5, random_state=b)

        Xy1 = np.concatenate((X1, y1.reshape((-1, 1))), 1)
        Xy2 = np.concatenate((X2, y2.reshape((-1, 1))), 1)
        n1 = Xy1.shape[0]
        n2 = Xy2.shape[0]

        term_1 = np.sum(pairwise_kernels(Xy1, metric="rbf", gamma=gamma))
        term_2 = np.sum(pairwise_kernels(Xy2, metric="rbf", gamma=gamma))
        term_12 = np.sum(pairwise_kernels(Xy1, Xy2, metric="rbf", gamma=gamma))

        mmd_null[b] = (
            (1 / (n1**2)) * term_1
            + (1 / (n2**2)) * term_2
            - 2 * (1 / (n1)) * (1 / (n2)) * term_12
        )

    pv = (1 + len(np.where(mmd_null > mmd)[0])) / (1 + n_boot)

    return pv
