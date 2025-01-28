import numpy as np
import pandas as pd
import os
import re
import argparse
from tqdm import tqdm
import copy
from utils import *
from datasets import DatasetWrapper
from tests.cdd import cdd_test
from tests.mmd import mmd_test
from tests.kl_boot import kl_boot_test
from tests.kl_samp import kl_samp_test
from tests.kl_dr import kl_dr_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", type=str, choices=["kl_boot", "kl_samp", "kl_dr", "cdd", "mmd"]
    )
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--ds_cov", type=float, default=0.0)
    parser.add_argument("--exp_dir", type=str)

    parser.add_argument("--attack_split", default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)

    parser.add_argument("--src_split", type=float, default=0.5)
    parser.add_argument("--sub_split", type=float, default=0.5)
    parser.add_argument("--n_bootstrap", type=int, default=200)
    parser.add_argument("--n_simulations", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eps", default=1e-8)

    # Random Forest / Decision Tree parameters
    parser.add_argument("--rf_n_estimators", type=int, default=None)  # 100)
    parser.add_argument("--min_samples_split", type=int, default=None)  # 25)
    parser.add_argument("--max_depth", type=int, default=None)  # 7)
    parser.add_argument("--max_features", type=int, default=None)  # 2)
    parser.add_argument("--criterion", type=str, default=None)
    # KNN parameters
    parser.add_argument("--knn_n_neighbors", default=None)
    # cdd parameters
    parser.add_argument("--gamma", default=None)
    args = parser.parse_args()

    print("=" * 40)
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")

    os.makedirs(args.exp_dir, exist_ok=True)
    if args.attack_split is None:
        attack_splits = [
            float(re.findall("\d+\.\d+", fname)[0])
            for fname in sorted(os.listdir(args.exp_dir))
        ]
    else:
        attack_splits = [float(args.attack_split)]

    ## check if 'seeds' exists
    seed_dir = f"{args.exp_dir}/../seeds"
    if not os.path.exists(seed_dir):
        os.makedirs(seed_dir)
        generate_all_seeds(seed_dir)

    # get the test function and corresponding keyword arguments
    test_func, method_kwargs = process_test_method(args)
    for attack_split in attack_splits:
        print(f"a_s: {attack_split}")

        # load/generate seeds for data split
        seeds = get_seeds(seed_dir, attack_split)

        # resulting p-value array
        p_values = np.zeros((args.n_simulations,))

        for i in tqdm(range(args.n_simulations)):
            # Dataset preparation
            ds = DatasetWrapper(
                args.dataset,
                attack_split,
                args.src_split,
                args.sub_split,
                seeds[i],
                s=args.ds_cov,
            )
            # Call the test method
            p_values[i] = test_func(
                *ds.source,
                *ds.target,
                n_boot=args.n_bootstrap,
                seed=seeds[i],
                **method_kwargs,
            )

        # Save p-values
        p_value_path = f"{args.exp_dir}/p_values_{args.dataset}_{attack_split:.2f}_{args.method}.npy"
        write_array(p_values, p_value_path)

        n_rejected = (p_values < args.alpha).sum()
        perc_rejected = ((n_rejected / len(p_values)) * 100.0).astype(int)

        print(f"rejected: {perc_rejected}%, failed to reject: {100-perc_rejected}%")


def process_model_params(args: argparse.Namespace):
    model_params = {
        "name": args.model,
        "device": "cuda" if args.model in ["vgg16", "cnn"] else None,
        "epochs": args.n_epochs,
        "batch_size": args.batch_size,
    }

    if args.model in ["rf", "dt"]:
        if args.rf_n_estimators is not None:
            model_params["n_estimators"] = args.rf_n_estimators
        if args.min_samples_split is not None:
            model_params["min_samples_split"] = args.min_samples_split
        if args.max_depth is not None:
            model_params["max_depth"] = args.max_depth
        if args.max_features is not None:
            model_params["max_features"] = args.max_features
        if args.criterion is not None:
            model_params["criterion"] = args.criterion
    elif args.model == "knn":
        model_params["n_neighbors"] = int(args.knn_n_neighbors)
    elif args.model == "log_reg":
        model_params["solver"] = "liblinear"

    return model_params


def process_test_method(args: argparse.Namespace):
    model_params = process_model_params(args)

    kwargs = {}
    if args.method.startswith("kl"):
        kwargs["model_params"] = model_params
    elif args.method == "mmd":
        kwargs["gamma"] = args.gamma

    test_func = {
        "kl_boot": kl_boot_test,
        "kl_samp": kl_samp_test,
        "kl_dr": kl_dr_test,
        "cdd": cdd_test,
        "mmd": mmd_test,
    }[args.method]

    return test_func, kwargs


def get_seeds(exp_dir, attack_split: float):
    seed_path = f"{exp_dir}/data_seeds_{attack_split:.2f}.npy"
    if os.path.exists(seed_path):
        seed_reps = load_array(seed_path)
        print("Seeds loaded!")
    else:
        seed_reps = np.random.randint(0, np.iinfo(np.int32).max, size=(100,))
        write_array(seed_reps, seed_path)
        print("Seeds generated and saved!")
    return seed_reps


def generate_all_seeds(seed_dir):
    print("Generating seeds...")
    a_s = [
        0.00,
        0.02,
        0.04,
        0.05,
        0.06,
        0.08,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    ]
    for attack_split in tqdm(a_s):
        seed_path = f"{seed_dir}/data_seeds_{attack_split:.2f}.npy"
        seed_reps = np.random.randint(0, np.iinfo(np.int32).max, size=(100,))
        write_array(seed_reps, seed_path)


if __name__ == "__main__":
    main()
