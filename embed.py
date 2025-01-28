import torch
from torchvision.models import resnet18
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm

torch.cuda.empty_cache()
from torchvision import models


class ResNetMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.loss = torch.nn.CrossEntropyLoss()

    # @auto_move_data
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.005)


class ResNetCIFAR10(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.model = resnet18(num_classes=10)
        self.model = resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        self.loss = torch.nn.CrossEntropyLoss()

    # @auto_move_data
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.005)


# model = resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
def main():
    # mnist
    trans = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                (0.5,),
                (0.5),
            ),
        ]
    )
    train_ds = MNIST("datasets/mnist", train=True, transform=trans)
    test_ds = MNIST("datasets/mnist", train=False, transform=trans)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64)

    model = ResNetMNIST()
    trainer = pl.Trainer(gpus=1, max_epochs=1, progress_bar_refresh_rate=20)
    trainer.fit(model, train_dl)
    model.eval()
    model.model.fc = torch.nn.Identity()

    train_dl = DataLoader(train_ds, batch_size=64)
    embs = []
    for batch in tqdm(iter(train_dl), total=len(train_dl)):
        x, y = batch
        preds = model(x)
        # preds, probs = get_pred(x, model)
        embs.append(preds.cpu())
    torch.save(torch.concat(embs), "mnist_train_embs.pt")

    embs = []
    for batch in tqdm(iter(test_dl), total=len(test_dl)):
        x, y = batch
        preds = model(x)
        # preds, probs = get_pred(x, model)
        embs.append(preds.cpu())
    torch.save(torch.concat(embs), "mnist_test_embs.pt")

    model.to("cpu")
    del model

    # cifar10
    trans = T.Compose(
        [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    train_ds = CIFAR10(
        "/home/itesliuk/energy_ood/data/cifarpy", train=True, transform=trans
    )
    test_ds = CIFAR10(
        "/home/itesliuk/energy_ood/data/cifarpy", train=False, transform=trans
    )
    train_dl = DataLoader(train_ds, batch_size=32)
    test_dl = DataLoader(test_ds, batch_size=32)

    model = ResNetCIFAR10()
    model.eval()
    embs = []
    for i, batch in enumerate(tqdm(iter(train_dl), total=len(train_dl))):
        x, y = batch
        preds = model(x)
        # preds, probs = get_pred(x, model)
        embs.append(preds.detach().cpu())
        if i == len(train_dl) // 2:
            torch.save(torch.concat(embs), "cifar10_train_embs_v1.pt")
            embs = []
    torch.save(torch.concat(embs), "cifar10_train_embs_v2.pt")

    embs = []
    for batch in tqdm(iter(test_dl), total=len(test_dl)):
        x, y = batch
        preds = model(x)
        # preds, probs = get_pred(x, model)
        embs.append(preds.detach().cpu())
    torch.save(torch.concat(embs), "cifar10_test_embs.pt")


import numpy as np
from sklearn.feature_selection import mutual_info_classif


def mi_filter(X, y, pmax=50):
    mi = np.zeros(X.shape[1])
    for j in tqdm(np.arange(X.shape[1])):
        mi[j] = mutual_info_classif(X[:, j].reshape(-1, 1), y)
    sel = np.argsort(-mi)[0:pmax]
    return sel


def preprocess():
    mnist_train_x = torch.load("mnist_train_embs.pt").numpy()
    mnist_train_y = MNIST("datasets/mnist", train=True).targets.numpy()
    mnist_test = torch.load("mnist_test_embs.pt").numpy()
    mnist_test_y = MNIST("datasets/mnist", train=False).targets.numpy()
    cifar_train = torch.load("cifar10_train_embs.pt").numpy()
    cifar_train_y = np.array(
        CIFAR10("/home/itesliuk/energy_ood/data/cifarpy", train=True).targets
    )
    cifar_test = torch.load("cifar10_test_embs.pt").numpy()
    cifar_test_y = np.array(
        CIFAR10("/home/itesliuk/energy_ood/data/cifarpy", train=False).targets
    )

    def label_tranform_mnist(y):
        y_copy = y.copy()
        y_copy[y % 2 == 0] = 1
        y_copy[y % 2 != 0] = 0
        return y_copy

    def label_tranform_cifar10(y):
        y_copy = y.copy()
        y_copy = np.zeros_like(y)
        y_copy[(y == 0) | (y == 1) | (y == 8) | (y == 9)] = 1
        # y_copy[y%2 != 0] = 0
        return y_copy

    def get_mnist(x, y):
        mtr_sel = mi_filter(x, y, 50)
        return np.hstack([x[:, mtr_sel], label_tranform_mnist(y).reshape(-1, 1)])

    def get_cifar10(x, y):
        mtr_sel = mi_filter(x, y, 50)
        return np.hstack([x[:, mtr_sel], label_tranform_cifar10(y).reshape(-1, 1)])

    mnist_train_sel = get_mnist(mnist_train_x, mnist_train_y)
    mnist_test_sel = get_mnist(mnist_test, mnist_test_y)
    np.save("mnist_train_embs_50.npy", mnist_train_sel)
    np.save("mnist_test_embs_50.npy", mnist_test_sel)

    cifar_train_sel = get_cifar10(cifar_train, cifar_train_y)
    cifar_test_sel = get_cifar10(cifar_test, cifar_test_y)
    np.save("cifar10_train_embs_50.npy", cifar_train_sel)
    np.save("cifat10_test_embs_50.npy", cifar_test_sel)


if __name__ == "__main__":
    main()
