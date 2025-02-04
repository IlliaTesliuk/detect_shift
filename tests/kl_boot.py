import numpy as np
from typing import Literal, Optional
from models import ModelWrapper
from utils import data_split, compute_kl_divergence, bootstrap_sample, get_model_params


# Method 1
def kl_boot_test(
    X_src: np.array,
    y_src: np.array,
    X_tgt: np.array,
    y_tgt: np.array,
    n_boot: int,
    model_params: dict,
    seed: Optional[int] = None,
):
    # dataset
    src1, src2 = data_split(X_src, y_src, 0.5, seed)
    tgt1, tgt2 = data_split(X_tgt, y_tgt, 0.5, seed)

    # model
    # print(model_params)
    model_params, model_kwargs = get_model_params(model_params)
    # print(model_params)
    # print(model_kwargs)
    model_name = model_params.name
    device = model_params.device
    epochs = model_params.epochs
    batch_size = model_params.batch_size

    # Train model on S1
    m_s1 = ModelWrapper(model_name, device, **model_kwargs)
    m_s1.train(src1, epochs, batch_size)
    # Inference on T2
    q_s1_t2 = m_s1.predict(tgt2, batch_size)

    # Train model on T1
    m_t1 = ModelWrapper(model_name, device, **model_kwargs)
    m_t1.train(tgt1, epochs, batch_size)
    # Inference on T2
    q_t1_t2 = m_t1.predict(tgt2, batch_size)

    # Compute W_hat
    W_hat = compute_kl_divergence(q_s1_t2, q_t1_t2)
    W_hat_null = np.zeros(n_boot)

    for b in range(n_boot):
        # Bootstrap sample
        src2_boot = bootstrap_sample(src2)

        # Train on S2_boot
        m_s2_boot = ModelWrapper(model_name, device, **model_kwargs)
        m_s2_boot.train(src2_boot, epochs, batch_size)

        # Inference on T2
        q_s2_boot_t2 = m_s2_boot.predict(tgt2, batch_size)

        # Compute W_hat_null
        W_hat_null[b] = compute_kl_divergence(q_s1_t2, q_s2_boot_t2)

    # Compute p-value
    p_value = (1 + np.sum(W_hat_null > W_hat)) / (1 + n_boot)

    return p_value
