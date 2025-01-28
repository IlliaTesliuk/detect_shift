import os
import torch
import time
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms, models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class ModelWrapper:
    def __init__(self, model_type, device=None, **kwargs):
        self.device = torch.device(device) if device is not None else None
        self.model = self._load_model(model_type, **kwargs)
        self.is_scikit = "sklearn" in str(type(self.model))
        self.train_history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

    def _load_model(self, model_type, **kwargs):
        if model_type == "vgg":
            model = VGG16Binary(self.device)
        elif model_type == "cnn":
            model = CustomCNN(self.device)
        elif model_type == "log_reg":
            model = LogisticRegression(**kwargs)
        elif model_type == "sgd":
            model = SGDClassifier(loss="log_loss", **kwargs)
        elif model_type == "dt":
            model = DecisionTreeClassifier(**kwargs)
        elif model_type == "rf":
            model = RandomForestClassifier(**kwargs)
        elif model_type == "knn":
            model = KNeighborsClassifier(**kwargs)
        return model

    def train(self, train_data, epochs=None, batch_size=32, partial=False):
        if self.is_scikit:
            data, targets = train_data
            if partial:
                # print(len(targets))
                self.model.partial_fit(data, targets, classes=[0, 1])
            else:
                self.model.fit(data, targets)
            return

        # os.makedirs(model_dir, exist_ok=True)
        # ckpt_path = f"{model_dir}/model_{model_name}_{epochs-1}.pt"
        # if os.path.exists(ckpt_path):
        #    model.load_state_dict(torch.load(ckpt_path))
        #    model.eval()
        #    print(f"Loaded model {model_name}!")
        #    return model

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        start_time_sec = time.time()
        for epoch in range(1, epochs + 1):
            print(f">> Epoch {epoch} <<")

            # 1. Train
            self.model.train()
            num_train_correct, num_train_examples = 0, 0
            train_loss = 0.0

            for batch in tqdm(train_loader):
                optimizer.zero_grad()

                x = batch[0].to(self.device)
                y = batch[1].to(self.device)

                yhat = self.model(x)

                loss = loss_fn(yhat, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.data.item() * x.size(0)
                num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                num_train_examples += x.shape[0]

            train_acc = num_train_correct / num_train_examples
            train_loss = train_loss / len(train_loader.dataset)

            self.train_history["loss"].append(train_loss)
            # history['val_loss'].append(val_loss)
            self.train_history["acc"].append(train_acc)
            # history['val_acc'].append(val_acc)

        # end of training loop
        end_time_sec = time.time()
        total_time_sec = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / epochs
        print()
        print("Time total:     %5.2f sec" % (total_time_sec))
        print("Time per epoch: %5.2f sec" % (time_per_epoch_sec))

        # if save:
        #    # saving training metrics
        ##    with open(f"{model_dir}/model_{model_name}_{epochs-1}.json","w") as fout:
        #       json.dump(history, fout)
        #    # saving model
        #    torch.save(model.state_dict(), ckpt_path)
        self.model.eval()

        return

    def predict(self, data, batch_size=32):
        if self.is_scikit:
            X, y = data
            pred = self.model.predict_proba(
                X,
            )[:, 1]
            return pred

        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        pred = np.zeros((len(data), len(data.classes)), dtype=np.float64)
        # print(pred.shape)
        for i, data in enumerate(tqdm(dataloader)):
            x = data[0] if len(data) > 1 else data
            x = x.to(self.model.device)

            yhat = self.model(x).cpu().detach().numpy()
            pred[i * batch_size : (i + 1) * batch_size, :] = yhat
        return pred


class VGG16Binary(torch.nn.Module):
    def __init__(self, device):
        super(VGG16Binary, self).__init__()

        model1 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in model1.parameters():
            param.requires_grad = False
        model1.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        model1.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 2),
            torch.nn.Sigmoid(),
        )
        self.model = model1

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

    def forward(self, x):
        y = self.model(x)
        return y


class CustomCNN(torch.nn.Module):
    def __init__(self, device):
        super(CustomCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 2)
        self.sigmoid = torch.nn.Sigmoid()

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = self.sigmoid(x)
        return output
