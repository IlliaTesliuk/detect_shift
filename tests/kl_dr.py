import numpy as np
from typing import Literal, Optional
from models import ModelWrapper
from utils import data_split, prepare_D_data, get_model_params


# Method 3
def kl_dr_test(
    X_src: np.array,
    y_src: np.array,
    X_tgt: np.array,
    y_tgt: np.array,
    n_boot: int,
    model_params: dict,
    seed: Optional[int] = None,
    eps: float = 1e-8,
):
    # Dataset
    src1, src2 = data_split(X_src, y_src, 0.5, seed)
    tgt1, tgt2 = data_split(X_tgt, y_tgt, 0.5, seed)
    d_train, d_test1, n_z0_train, n_z1_train = prepare_D_data(
        src1, src2, tgt1, tgt2, seed
    )

    # Model
    model_params, model_kwargs = get_model_params(model_params)
    model_name = model_params.name
    device = model_params.device
    epochs = model_params.epochs
    batch_size = model_params.batch_size

    # Train on ([S1,T],Z)
    m_main = ModelWrapper(model_name, device, **model_kwargs)
    m_main.train(d_train, epochs, batch_size)
    # Inference on D_Te1 (probabilities for z)
    proba_m_main = m_main.predict(d_test1, batch_size)

    p = np.clip(proba_m_main, eps, None)
    q = np.clip(1 - proba_m_main, eps, None)
    r = np.log((p / q) * (n_z0_train / n_z1_train))

    # Compute W_hat
    W_hat = np.mean(r)
    W_hat_null = np.zeros(n_boot)

    # Train model on S2
    m_s2 = ModelWrapper(model_name, device, **model_kwargs)
    m_s2.train(src2, epochs, batch_size)
    # Inference on D_test 1
    X_D_te1, y_D_te1 = d_test1[0][:, :-1], d_test1[0][:, -1]
    proba_model_temp = m_s2.predict((X_D_te1, y_D_te1), batch_size)

    for b in range(n_boot):
        # Create sample
        y_null_b = np.random.binomial(1, proba_model_temp, size=len(y_D_te1))

        Xy_D_test1_b, Z_D_test1_b = d_test1
        Xy_D_test1_b[:, -1] = y_null_b

        # Inference on XY_D_test1_b
        prob_model_main_b = m_main.predict((Xy_D_test1_b, Z_D_test1_b), batch_size)

        p = np.clip(prob_model_main_b, eps, None)
        q = np.clip(1 - prob_model_main_b, eps, None)
        r_b = np.log((p / q) * (n_z0_train / n_z1_train))

        W_hat_null[b] = np.mean(r_b)

    p_value = (1 + np.sum(W_hat_null > W_hat)) / (1 + n_boot)

    return p_value
