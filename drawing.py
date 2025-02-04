import os
import re
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import scienceplots
from utils import load_w_hat, calculate_p_value, load_array

BOX_PROPS = dict(boxstyle="round", facecolor="white", alpha=0.5)
with plt.style.context("science"):
    SCOLORS = {
        "red": sns.color_palette()[3],
        "green": sns.color_palette()[1],
        "blue": sns.color_palette()[0],
        "yellow": sns.color_palette()[2],
        "purple": sns.color_palette()[4],
        "gray": sns.color_palette()[6],
        "black": sns.color_palette()[5],
    }

from utils import load_array
import numpy as np


def load_pvalues_triples(exp_dir):
    fnames = sorted(os.listdir(f"{exp_dir}"))[:]
    ss1, ss2, ss3 = [], [], []
    pvalues1, pvalues2, pvalues3 = [], [], []
    for fname in fnames:
        if fname.startswith("p_values"):
            arr = load_array(f"{exp_dir}/{fname}")
            f = float(fname[fname.rfind("_") + 1 : fname.rfind(".")])
            if np.all(arr[0, :] == 0.0) and np.all(arr[1, :] == 0.0):
                ss3.append(f)
                pvalues3.append(arr[2, :])
            else:
                ss1.append(f)
                ss2.append(f)
                ss3.append(f)
                pvalues1.append(arr[0, :])
                pvalues2.append(arr[1, :])
                pvalues3.append(arr[2, :])

    pv1 = np.array(pvalues1)
    pv2 = np.array(pvalues2)
    pv3 = np.array(pvalues3)

    # if 1.0 not in ss:
    #    ss.append(1.0)
    #    pv = np.concatenate([pv,np.full((1,3,100),0.00497512)])
    # return ss, pv
    n_rejected1 = np.sum(pv1 < 0.05, axis=1)
    perc_rejected1 = n_rejected1 / 100
    n_rejected2 = np.sum(pv2 < 0.05, axis=1)
    perc_rejected2 = n_rejected2 / 100
    n_rejected3 = np.sum(pv3 < 0.05, axis=1)
    perc_rejected3 = n_rejected3 / 100
    return ss1, ss2, ss3, perc_rejected1, perc_rejected2, perc_rejected3


def load_pvalues_single(
    exp_dir,
    dataset,
    s=None,
    n="cdd",
):
    if s is None:
        s = sorted(
            float(re.findall("\d+\.\d+", f)[0])
            for f in os.listdir(f"{exp_dir}")
            if f.startswith("p_values") and f.endswith(f"_{n}.npy")
        )
    ss = []
    pvalues = []
    for cur_s in s:
        fname = f"p_values_{dataset}_{cur_s:.2f}_{n}.npy"
        arr = load_array(f"{exp_dir}/{fname}")
        ss.append(cur_s)
        # ss.append(float(fname.split("_")[-2]))
        pvalues.append(arr)

    pv = np.array(pvalues)
    n_rejected = np.sum(pv < 0.05, axis=1)
    perc_rejected = n_rejected / 100
    return ss, perc_rejected


def process_all(ss1, ss2, ss3, pc1, pc2, pc3, title, save_path=None):
    with plt.style.context("science"):
        plt.rcParams.update({"font.size": 16})

        fig = plt.figure(figsize=(8, 5))

        plt.plot(ss1, pc1, "r-")
        plt.plot(ss2, pc2, "g-")
        plt.plot(ss3, pc3, "b-")
        plt.hlines(y=0.05, xmin=0.0, xmax=1.0, linestyles="dashdot", colors="k")

        legend = [f"M1 (DT): {pc1[0]:.2f}", f"MMD: {pc2[0]:.2f}", f"CDD: {pc3[0]:.2f}"]
        plt.legend(legend, loc="center right")

        plt.xlabel("s")
        plt.ylabel("n_rejected")
        plt.title(title)

        if save_path:
            plt.savefig(save_path)
            plt.close()
            plt.clf()
        else:
            plt.show()
        #


def plot_dataset(ss, pc, names, title, figsize=(8, 5), save_path=None):
    assert len(ss) == len(pc)
    assert len(pc) == len(names)
    n_methods = len(ss)
    colors = ["r-", "g-", "b-", "m-", "y-", "c-"]
    legend = []
    with plt.style.context("science"):
        plt.rcParams.update({"font.size": 16})

        fig = plt.figure(figsize=figsize)
        for i in range(n_methods):
            plt.plot(ss[i], pc[i], colors[i])
            legend.append(f"{names[i]}: {pc[i][0]:.2f}")

        plt.hlines(y=0.05, xmin=0.0, xmax=1.0, linestyles="dashdot", colors="k")
        plt.legend(legend, loc="center right")

        plt.xlabel(f"Percentage of Labels Flipped ($\%$)")
        # plt.xlabel(f"Label Flipping Attack Rate (%)")
        plt.ylabel(f"Rejection Rate of $H_0$ ($\%$)")
        # plt.ylabel(f"$H_0$ Rejection Percentage (%)")
        plt.title(title)
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)

        if save_path:
            plt.savefig(save_path)
            plt.close()
            plt.clf()
        else:
            plt.show()
        #


DATASETS = [
    "artificial_0.0",
    "artificial_0.1",
    "artificial_0.5",
    "adult_income",
    "bank_marketing",
    "credit_default",
    "credit_risk",
    "pulsars",
    "cifar10_binary",
    "mnist_binary",
]

METHOD_NAMES = {
    "kl_boot": "KL Boot",
    "kl_dr": "KL DR",
    "cdd": "CDD",
    "mmd": "MMD",
    "wrs": "WRS",
    "wrs_dt": "WRS"
}


def dataset_name(ds: str):
    if ds.startswith("artificial"):
        name = f"Artificial (s={ds[-3:]})"
    elif ds.endswith("binary"):
        name = ds.split("_")[0].upper() + " Binary"
    else:
        name = " ".join(ds.split("_")).title()
    return name


def process_experiments(exp_dir, methods, figsize=(8, 5), save=True):
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for dataset in DATASETS:
        print(dataset)
        ds_dir = os.path.join(exp_dir, dataset)

        ds_name = dataset_name(dataset)
        ds_save_name = "artificial" if dataset.startswith("artificial") else dataset
        attack_perc, p_values, m_names = [], [], []
        for method in methods:
            method_name = METHOD_NAMES[method]
            method_dir = os.path.join(ds_dir, method if not method.startswith("wrs") else "wrs")
            if os.path.exists(method_dir) and len(os.listdir(method_dir)) > 0:
                ap, pv = load_pvalues_single(method_dir, ds_save_name, None, method)

                attack_perc.append(ap)
                p_values.append(pv)
                m_names.append(method_name)

        plot_dataset(
            attack_perc,
            p_values,
            m_names,
            ds_name,
            figsize,
            save_path=f"{plots_dir}/{dataset}.png" if save else None,
        )
    draw_grid(plots_dir, True)


def draw_grid(exp_dir, save=False):
    img = Image.open(f"{exp_dir}/artificial_0.0.png")
    w, h = img.size
    print(w, h)
    grid = Image.new("RGB", [w * 4, h * 3], color=(255, 255, 255))
    grid.paste(img, [0, 0])
    grid.paste(Image.open(f"{exp_dir}/artificial_0.1.png"), [w, 0])
    grid.paste(Image.open(f"{exp_dir}/artificial_0.5.png"), [2 * w, 0])
    grid.paste(Image.open(f"{exp_dir}/adult_income.png"), [3 * w, 0])
    grid.paste(Image.open(f"{exp_dir}/bank_marketing.png"), [0, h])
    grid.paste(Image.open(f"{exp_dir}/credit_default.png"), [w, h])
    grid.paste(Image.open(f"{exp_dir}/credit_risk.png"), [2 * w, h])
    grid.paste(Image.open(f"{exp_dir}/pulsars.png"), [3 * w, h])
    grid.paste(Image.open(f"{exp_dir}/cifar10_binary.png"), [w, 2 * h])
    grid.paste(Image.open(f"{exp_dir}/mnist_binary.png"), [2 * w, 2 * h])
    if save:
        grid.save(f"{exp_dir}/grid.png")
        return
    return grid


def cur():
    ss1, _, _, pv1, _, _ = load_pvalues_triples("exps/results/adult_income/dt")
    ss_mmd, pv_mmd = load_pvalues_single(
        "exps/results/adult_income/mmd",
        "adult_income",
        [0.0, 0.02, 0.04, 0.5, 0.6, 0.7, 0.8, 1.0],
        "mmd",
    )
    ss_cdd, pv_cdd = load_pvalues_single(
        "exps/results/adult_income/cdd",
        "adult_income",
        [
            0.00,
            0.02,
            0.04,
            0.06,
            0.08,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.0,
        ],
        "cdd",
    )
    process_all(
        ss1,
        ss_mmd,
        ss_cdd,
        pv1,
        pv_mmd,
        pv_cdd,
        "Adult Income",
        "exps/results/plots/250120/adult_income.png",
    )

    ss1, _, _, pv1, _, _ = load_pvalues_triples("exps/results/bank_marketing/dt")
    ss_mmd, pv_mmd = load_pvalues_single(
        "exps/results/bank_marketing/mmd", "bank_marketing", [0.0, 0.5, 0.8, 1.0], "mmd"
    )
    ss_cdd, pv_cdd = load_pvalues_single(
        "exps/results/bank_marketing/cdd",
        "bank_marketing",
        [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 1.0],
        "cdd",
    )
    process_all(
        ss1,
        ss_mmd,
        ss_cdd,
        pv1,
        pv_mmd,
        pv_cdd,
        "Bank Marketing",
        "exps/results/plots/250120/bank_marketing.png",
    )

    ss1, _, _, pv1, _, _ = load_pvalues_triples("exps/results/credit_default/dt")
    ss_mmd, pv_mmd = load_pvalues_single(
        "exps/results/credit_default/mmd", "credit_default", [0.0, 0.5, 1.0], "mmd"
    )
    ss_cdd, pv_cdd = load_pvalues_single(
        "exps/results/credit_default/cdd",
        "credit_default",
        [0.00, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0],
        "cdd",
    )
    process_all(
        ss1,
        ss_mmd,
        ss_cdd,
        pv1,
        pv_mmd,
        pv_cdd,
        "Credit Default",
        "exps/results/plots/250120/credit_default.png",
    )

    ss1, _, _, pv1, _, _ = load_pvalues_triples("exps/results/credit_risk/dt")
    ss_mmd, pv_mmd = load_pvalues_single(
        "exps/results/credit_risk/mmd", "credit_risk", [0.0, 0.5, 1.0], "mmd"
    )
    ss_cdd, pv_cdd = load_pvalues_single(
        "exps/results/credit_risk/cdd",
        "credit_risk",
        [0.00, 0.02, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0],
        "cdd",
    )
    process_all(
        ss1,
        ss_mmd,
        ss_cdd,
        pv1,
        pv_mmd,
        pv_cdd,
        "Credit Risk",
        "exps/results/plots/250120/credit_risk.png",
    )

    ss1, _, _, pv1, _, _ = load_pvalues_triples("exps/results/custom/dt")
    ss_mmd, pv_mmd = load_pvalues_single(
        "exps/results/custom/mmd",
        "custom",
        [
            0.0,
            0.02,
            0.04,
            0.06,
            0.08,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
        ],
        "mmd",
    )
    ss_cdd, pv_cdd = load_pvalues_triples(
        "exps/results/custom/cdd",
        "custom",
        [
            0.0,
            0.02,
            0.04,
            0.06,
            0.08,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
        ],
        "cdd",
    )
    process_all(
        ss1,
        ss_mmd,
        ss_cdd,
        pv1,
        pv_mmd,
        pv_cdd,
        "Random Data",
        "exps/results/plots/250120/custom.png",
    )

    ss1, _, _, pv1, _, _ = load_pvalues_triples("exps/results/pulsars/dt")
    ss_mmd, pv_mmd = load_pvalues_single(
        "exps/results/pulsars/mmd", "pulsars", [0.0, 0.1, 0.2, 0.5, 1.0], "mmd"
    )
    ss_cdd, pv_cdd = load_pvalues_single(
        "exps/results/pulsars/cdd",
        "pulsars",
        [
            0.0,
            0.02,
            0.04,
            0.06,
            0.08,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
        ],
        "cdd",
    )
    process_all(
        ss1,
        ss_mmd,
        ss_cdd,
        pv1,
        pv_mmd,
        pv_cdd,
        "Pulsars",
        "exps/results/plots/250120/pulsars.png",
    )

    ss1, _, _, pv1, _, _ = load_pvalues_triples("exps/results/cifar10_binary/dt")
    ss_mmd, pv_mmd = load_pvalues_single(
        "exps/results/cifar10_binary/mmd",
        "cifar10_binary",
        [0.0, 0.2, 0.5, 0.8, 1.0],
        "mmd",
    )
    ss_cdd, pv_cdd = load_pvalues_single(
        "exps/results/cifar10_binary/cdd",
        "cifar10_binary",
        [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        "cdd",
    )
    process_all(
        ss1,
        ss_mmd,
        ss_cdd,
        pv1,
        pv_mmd,
        pv_cdd,
        "CIFAR10 (Binary)",
        "exps/results/plots/250120/cifar10_binary.png",
    )

    ss1, _, _, pv1, _, _ = load_pvalues_triples("exps/results/mnist_binary/dt")
    ss_mmd, pv_mmd = load_pvalues_single(
        "exps/results/mnist_binary/mmd", "mnist_binary", [0.0, 0.5, 1.0], "mmd"
    )
    ss_cdd, pv_cdd = load_pvalues_single(
        "exps/results/mnist_binary/cdd",
        "mnist_binary",
        [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        "cdd",
    )
    process_all(
        ss1,
        ss_mmd,
        ss_cdd,
        pv1,
        pv_mmd,
        pv_cdd,
        "MNIST (Binary)",
        "exps/results/plots/250120/mnist_binary.png",
    )

    from PIL import Image

    draw_grid("exps/results/plots/250120", True)


class Exp:
    def __init__(self, exp_dir, method: str, dataset: str, exp_num: int):
        self.method = method
        self.dataset = dataset
        self.exp_num = exp_num

        self.results_dir = os.path.join(exp_dir, "results")
        results = load_w_hat(self.results_dir)

        self.w_hat_null = results[0]
        self.w_hat = results[1]
        self.n_bootstrap = self.w_hat_null.shape[0]

        self.p_value = calculate_p_value(self.w_hat, self.w_hat_null, self.n_bootstrap)

    def latex_str(self):
        return "\n".join(
            (
                r"$\hat{W}=%.4f$" % (self.w_hat,),
                r"$p_{value}=%.4f$" % (self.p_value,),
                r"$n_{boot}=%d$" % (self.n_bootstrap,),
            )
        )

    def plot_exp(self):
        with plt.style.context("science"):
            plt.rcParams.update({"font.size": 16})
            fig = plt.figure(figsize=(8, 5))

            sns.kdeplot(self.w_hat_null, color=SCOLORS["blue"])

            ax = plt.gca()
            x = ax.lines[0].get_xydata()[:, 0]
            y = ax.lines[0].get_xydata()[:, 1]
            plt.vlines(
                x=self.w_hat,
                ymin=0,
                ymax=y.max() + 0.01,
                linewidth=2,
                colors=SCOLORS["red"],
            )

            x2_mask = x >= self.w_hat
            x2, y2 = x[x2_mask], y[x2_mask]
            ax.fill_between(x2, y2, color=SCOLORS["blue"], alpha=0.25)

            plt.legend(["$\hat{W}_{0}$", "$\hat{W}$"], frameon=True, loc="upper right")
            plt.title(
                f"Method {self.method}. {self.dataset} classification. Exp {self.exp_num}"
            )
            ax.text(
                0.73,
                0.6,
                self.latex_str(),
                transform=ax.transAxes,
                fontsize=16,
                verticalalignment="top",
                bbox=BOX_PROPS,
            )
            plt.show()
