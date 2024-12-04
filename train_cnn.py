import os.path

import torch
import torch.nn as nn
from dataset import AnimalDataset
from models import SimpleCNN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.datasets import CIFAR10
from tqdm.autonotebook import tqdm
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import shutil
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Train NN model")
    parser.add_argument("--data_path", "-d", type=str, default="data/animals", help="path to the dataset")
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--lr", "-l", type=float, default=1e-2)
    parser.add_argument("--log_path", "-p", type=str, default="tensorboard/animals")
    parser.add_argument("--checkpoint_path", "-c", type=str, default="trained_models/animals")
    args = parser.parse_args()

    return args


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="cool")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    # build model from scratch
    # model = SimpleCNN()
    # transfer learning
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(in_features=512, out_features=10)

    model.to(device)  # ~in-place function
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = CIFAR10(root=args.data_path, train=True, transform=transform, download=True)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=True,
        drop_last=False
    )
    test_dataset = CIFAR10(root=args.data_path, train=False, transform=transform, download=True)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=False,
        drop_last=False
    )
    # criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 2, 1, 1, 2, 1, 1, 1, 1, 1]).to(device))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    writer = SummaryWriter(args.log_path)
    best_acc = -100
    for epoch in range(args.epochs):
        # TRAIN
        model.train()
        progress_bar = tqdm(train_dataloader, colour="yellow")
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, args.epochs, loss))
            writer.add_scalar("Train/loss", loss, epoch * len(train_dataloader) + i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # VALIDATION
        model.eval()
        all_losses = []
        all_labels = []
        all_predictions = []
        with torch.no_grad():  # equivalent to # with torch.inference_mode():
            for i, (images, labels) in enumerate(test_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                # _, predictions = torch.max(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                loss = criterion(output, labels)
                all_losses.append(loss.item())
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
            loss = np.mean(all_losses)
            accuracy = accuracy_score(all_labels, all_predictions)
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            print("Epoch {}/{}. Loss {:0.4f}. Acc {:0.4f}".format(epoch + 1, args.epochs, loss, accuracy))
            writer.add_scalar("Test/loss", loss, epoch)
            writer.add_scalar("Test/Accuracy", accuracy, epoch)
            plot_confusion_matrix(writer, conf_matrix, categories, epoch)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_path, "last.pt"))
            if accuracy > best_acc:
                torch.save(checkpoint, os.path.join(args.checkpoint_path, "best.pt"))
                best_acc = accuracy




if __name__ == '__main__':
    args = get_args()
    train(args)
