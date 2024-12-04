import torch
from pandas import pivot
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from PIL import Image
from tqdm.autonotebook import tqdm
import argparse
import os
import numpy as np
import cv2
from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser(description="Train DeeplabV3 model")
    parser.add_argument("--image_path", "-i", type=str, default="C:/Users/tam/Desktop/pythonProject/DL ML/DL/DL/scripts/my_pascal_voc/VOCdevkit/VOC2012/JPEGImages/2007_002619.jpg")
    parser.add_argument("--saved_checkpoint", "-o", type=str, default="trained_models_deeplab/best.pt", help="Continue from this checkpoint")
    args = parser.parse_args()
    return args


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.saved_checkpoint, map_location="cpu", weights_only=False)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.float()
    input_image = Image.open(args.image_path)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    start_epoch = checkpoint["epoch"]
    print(f"Start epoch: {start_epoch}")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(input_batch.to(device))['out'][0]
    output_predictions = output.argmax(0)

    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(input_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(r)
    axes[1].set_title("Segmentation Result")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    r.save("segmentation_result.png")

def cout(a):
    print(a)
    print(type(a))

if __name__ == '__main__':
    args = get_args()
    test(args)
