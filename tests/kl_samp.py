import numpy as np
from typing import Literal, Optional
from models import ModelWrapper
from utils import data_split, compute_kl_divergence, get_model_params


# Method 2
def kl_samp_test(
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
    model_params, model_kwargs = get_model_params(model_params)
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

    # Train model on S2
    m_s2 = ModelWrapper(model_name, device, **model_kwargs)
    m_s2.train(src2, epochs, batch_size)
    # Inference on T1
    q_s2_t1 = m_s2.predict(tgt1, batch_size)

    for b in range(n_boot):
        # Create sample
        z = np.random.binomial(1, q_s2_t1, size=len(tgt1[1]))
        tgt1_null = (tgt1[0], z)

        # Train on T1_null
        m_t1_null = ModelWrapper(model_name, device, **model_kwargs)
        m_t1_null.train(tgt1_null, epochs, batch_size)
        # Inference on T2
        q_t1_null_t2 = m_t1_null.predict(tgt2, batch_size)

        # Compute W_hat_null
        W_hat_null[b] = compute_kl_divergence(q_s1_t2, q_t1_null_t2)

    # Compute p-value
    p_value = (1 + np.sum(W_hat_null > W_hat)) / (1 + n_boot)

    return p_value
