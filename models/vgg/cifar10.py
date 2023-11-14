import torch
import numpy as np
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import time
import argparse
import matplotlib.pyplot as plt
from rich.progress import track

from model import VGG


class CIFAR10:
    model_options = ["VGG11", "VGG13", "VGG16", "VGG19"]

    config = {
        "epochs": 20,
        "batch_size": 256,
        "learn_ratio": 1e-2,
        "valid_ratio": 0.2,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_path": "./CIFAR10_model.ckpt",
        "plot": True,
        "model_name": "VGG13"
    }

    def __init__(self, cfg):
        self.cfg = cfg

        # define preprocessing of training dataset and validation dataset
        train_tran = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])

        # define preprocessing of test dataset
        test_tran = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ])

        # main dataset including training dataset and validation dataset
        main_data = torchvision.datasets.CIFAR10("./dataset", True,
                                                 transform=train_tran, download=True)
        # test dataset
        self.test_data = torchvision.datasets.CIFAR10("./dataset", False,
                                                      transform=test_tran, download=True)

        train_len = int(len(main_data) * (1 - self.cfg["learn_ratio"]))

        # split main_data into train dataset and valid dataset
        self.train_data, self.valid_data = torch.utils.data.random_split(dataset=main_data,
                                                                         lengths=[train_len,
                                                                                  len(main_data) - train_len],
                                                                         generator=torch.Generator())

        del main_data  # release useless variable

        self.train_dataLoader = DataLoader(self.train_data, batch_size=self.cfg["batch_size"], shuffle=True)
        self.valid_dataLoader = DataLoader(self.valid_data, batch_size=self.cfg["batch_size"], shuffle=True)
        self.test_dataLoader = DataLoader(self.test_data, batch_size=self.cfg["batch_size"], shuffle=False)

        self.model = VGG(3, 10, self.cfg["model_name"]).to(self.cfg["device"])

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg["learn_ratio"], momentum=0.9,
                                         weight_decay=5e-3)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                              milestones=[int(self.cfg["epochs"] * 0.48),
                                                                          int(self.cfg["epochs"] * 0.72)],
                                                              gamma=0.1, last_epoch=-1)

    def train(self):
        best_epoch = 0
        best_acc = 0.0
        history = []

        for epoch in range(self.cfg["epochs"]):

            train_acc, valid_acc = 0.0, 0.0
            train_loss, valid_loss = 0.0, 0.0

            self.model.train()
            for images, targets in track(self.train_dataLoader, description="train: {}".format(epoch + 1)):
                self.optimizer.zero_grad()
                images, targets = images.to(self.cfg["device"]), targets.to(self.cfg["device"])
                preds = self.model(images)
                loss = self.criterion(preds, targets)
                loss.backward()
                self.optimizer.step()

                _, pos = torch.max(preds, 1)
                train_acc += (pos.detach() == targets.detach()).sum().item()
                train_loss += loss.detach().item() * images.shape[0]

            self.model.eval()
            for images, targets in track(self.valid_dataLoader, description="valid: {}".format(epoch + 1)):
                images, targets = images.to(self.cfg["device"]), targets.to(self.cfg["device"])
                with torch.no_grad():
                    preds = self.model(images)
                    loss = self.criterion(preds, targets)
                    _, pos = torch.max(preds, 1)
                    valid_acc += (pos.detach() == targets.detach()).sum().item()
                    valid_loss += loss.detach().item() * images.shape[0]

            self.scheduler.step()

            train_acc, train_loss = train_acc / len(self.train_data), train_loss / len(self.train_data)
            valid_acc, valid_loss = valid_acc / len(self.valid_data), valid_loss / len(self.valid_data)

            if valid_acc > best_acc:
                best_epoch = epoch + 1
                best_acc = valid_acc
                torch.save(self.model.state_dict(), self.cfg["save_path"])

            print(
                "Training: loss: {:.5f}, accuracy: {:.5f}%,"
                "\nValidation: loss: {:.5f}, accuracy: {:.5f}%".format(
                    train_loss, train_acc * 100, valid_loss, valid_acc * 100))

            print("The best validation accuracy is : {:.5f}% at epoch {}".format(best_acc * 100, best_epoch))

            history.append([train_acc, train_loss, valid_acc, valid_loss])

        return history

    def test(self):
        test_acc = 0.0
        with torch.no_grad():
            for images, targets in track(self.test_dataLoader, description="final test:"):
                images, targets = images.to(self.cfg["device"]), targets.to(self.cfg["device"])
                preds = self.model(images)
                _, pos = torch.max(preds, 1)
                test_acc += (pos.detach() == targets.detach()).sum().item()

        print("Test Acc: {}%".format((test_acc / len(self.test_data)) * 100))

    def plot(self, history):
        if self.cfg["plot"] is False:
            return

        pre = str(int(time.time())) + '_' + self.cfg["model_name"]

        history = np.array(history)

        plt.figure(figsize=(10, 10))
        plt.plot(history[:, 1::2])
        plt.legend(['Train Loss', 'Valid Loss'])
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss Value')

        plt.xticks(np.arange(0, self.cfg["epochs"] + 1, step=10))
        plt.yticks(np.arange(0, 2.0, 0.1))
        plt.grid()
        plt.savefig(pre + '_loss_curve.png')
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.plot(history[:, 0::2])
        plt.legend(['Train Accuracy', 'Valid Accuracy'])
        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy')

        plt.xticks(np.arange(0, self.cfg["epochs"] + 1, step=10))
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.grid()
        plt.savefig(pre + '_accuracy_curve.png')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Start Training CIFAR10")
    parser.add_argument("--epochs", "-e", type=int, default=CIFAR10.config["epochs"],
                        help="training epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=CIFAR10.config["batch_size"],
                        help="batch size")
    parser.add_argument("--learn_ratio", "--lr", type=float, default=CIFAR10.config["learn_ratio"],
                        help="learning rate(it is gradually decayed by scheduler)")
    parser.add_argument("--valid_ratio", "--vr", type=float, default=CIFAR10.config["valid_ratio"],
                        help="validation dataset ratio")
    parser.add_argument("--device", "-d", type=str, default=CIFAR10.config["device"],
                        help="storage device name")
    parser.add_argument("--save_path", "--sp", type=str, default=CIFAR10.config["save_path"],
                        help="path to store the best model")
    parser.add_argument("--plot", "-p", type=bool, default=CIFAR10.config["plot"],
                        help="whether to plot training curve")
    parser.add_argument("--model_name", "-m", type=str, default=CIFAR10.config["model_name"],
                        help="select the available models: " + ",".join(CIFAR10.model_options))

    example = CIFAR10(vars(parser.parse_args()))
    example.plot(example.train())
    example.test()


if __name__ == "__main__":
    main()
