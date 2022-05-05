import torch as th
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as opt
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
import os
from model import YOLOv1
from tqdm import tqdm
from typing import Tuple, List

# Model Hyperparameters
S = 7
B = 2
C = 1000

# Tranforms Hyperparameters
RESIZE_D = 256
INPUT_D = 224

# Data Loading Hyperparameters
MINI_BATCH = 256
NUM_WORKERS = 5
SHUFFLE = True
PIN_MEMORY = True
DROP_LAST = True

# Training Hyperparameters
MAX_EPOCHS = 90
INIT_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
PATIENCE = 2
MIN_LR = 0.0001

# ImageNet Dataset Directory
IMAGENET_DIR_PATH = "/home/soul/Development/datasets/ImageNet"

# Compute Device (use a GPU if available)
DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

# Checkpoint Hyperparameters
TRAIN_MODEL = False # When the model is not trained, the final pretrained model weights are loaded for the model to be evaluated.
LOAD_MODEL = True   # When the model is trained, it can either be trained from scratch or from a checkpoint.
CHECKPOINT_PATH = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                  "Detection/checkpoints/pretrain_checkpoint.pt"
PRETRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                           "Detection/checkpoints/pretrained_model_weights.pt"
CHECKPOINT_T = 1


def train_epoch(train_loader: DataLoader,
                model: YOLOv1,
                optimizer: opt.SGD,
                criterion: nn.CrossEntropyLoss) -> float:
    """
    Train the YOLO model for one epoch and return the average loss per training instance for this epoch.

    :param train_loader: The DataLoader of the ImageNet train set
    :param model: The YOLOv1 classification model
    :param optimizer: The SGD with momentum optimizer
    :param criterion: The cross-entropy loss criterion
    :return: The average training loss per instance for this epoch
    """
    av_loss = 0.
    model.train()
    for x, y_gt in train_loader:
        x, y_gt = x.to(DEVICE), y_gt.to(DEVICE)
        y_pred = model(x)

        loss = criterion(y_pred, y_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        av_loss += loss.item()

    av_loss /= len(train_loader)
    return av_loss


def validate_epoch(val_loader: DataLoader,
                   model: YOLOv1,
                   criterion: nn.CrossEntropyLoss) -> float:
    """
    Validate the YOLO model and return the average loss per validation instance for this epoch.

    :param val_loader: The DataLoader of the ImageNet validation set
    :param model: The YOLOv1 classification model
    :param criterion: The cross-entropy loss criterion
    :return: The average validation loss per instance for this epoch
    """
    av_loss = 0.
    with th.no_grad():
        model.eval()
        for x, y_gt in val_loader:
            x, y_gt = x.to(DEVICE), y_gt.to(DEVICE)
            y_pred = model(x)
            loss = criterion(y_pred, y_gt)
            av_loss += loss.item()

    av_loss /= len(val_loader)
    return av_loss


def train(train_loader: DataLoader,
          val_loader: DataLoader,
          model: YOLOv1,
          optimizer: opt.SGD,
          scheduler: lr_scheduler.ReduceLROnPlateau,
          criterion: nn.CrossEntropyLoss,
          epoch: int,
          train_loss_history: List,
          val_loss_history: List) -> None:
    """
    Train the YOLOv1 model for a MAX_EPOCHS number of epochs. If the validation loss does not decrease, scale the
    learning rate by 0.1. Plot a bar to illustrate the progress of training. Every CHECKPOINT_T epochs, create a new
    checkpoint.

    :param train_loader: The DataLoader of the ImageNet train set
    :param val_loader: The DataLoader of the ImageNet validation set
    :param model: The YOLOv1 classification model
    :param optimizer: The SGD with momentum optimizer
    :param scheduler:
    :param criterion: The cross-entropy loss criterion
    :param epoch: The starting epoch of the training
    :param train_loss_history: The history of the training losses up to the current epoch
    :param val_loss_history: The history of the validation losses up to the current epoch
    """

    pbar = tqdm(total=MAX_EPOCHS, desc='Training Epoch', initial=epoch, unit='epoch')
    while epoch < MAX_EPOCHS:
        epoch += 1

        train_loss = train_epoch(train_loader, model, optimizer, criterion)
        val_loss = validate_epoch(val_loader, model, criterion)
        scheduler.step(val_loss)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if epoch % CHECKPOINT_T == 0:
            th.save({'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(),
                     'train_loss_history': train_loss_history,
                     'val_loss_history': val_loss_history
                     }, CHECKPOINT_PATH)

        pbar.set_postfix_str(f'Train Loss={train_loss:.3f}, Val Loss={val_loss:.3f}')
        pbar.update(1)
    pbar.close()


def measure_accuracy(model: YOLOv1, val_loader: DataLoader) -> Tuple[float, float]:
    """
    Calculate and return the single-crop top1 and top5 accuracies of the model on the validation set of the ImageNet
    dataset.

    :param model: The YOLOv1 classification model
    :param val_loader: The DataLoader of the ImageNet validation set
    :return: The single-crop top1 and top5 accuracies
    """
    top1_accuracy = 0.
    top5_accuracy = 0.
    total = 0
    with th.no_grad():
        model.eval()
        for x, y_gt in val_loader:
            x, y_gt = x.to(DEVICE), y_gt.to(DEVICE)
            y_pred = model(x)

            _, top5_preds = th.topk(y_pred, k=5, dim=-1)
            top1_accuracy += th.sum(top5_preds[:, 0] == y_gt).item()
            top5_accuracy += th.sum(top5_preds == y_gt.reshape(-1, 1)).item()
            total += x.shape[0]

        top1_accuracy /= total
        top5_accuracy /= total

        return top1_accuracy, top5_accuracy


def setup_train() -> Tuple[DataLoader,
                           DataLoader,
                           YOLOv1,
                           opt.SGD,
                           lr_scheduler.ReduceLROnPlateau,
                           nn.CrossEntropyLoss]:
    """
    Instantiate the model, the optimizer, the learning rate scheduler, the loss criterion, the train and the validation
    ImageNet datasets and their corresponding loaders.

    :return: The DataLoader objects of the training and the validation ImageNet dataset, the YOLO classification model,
             an optimizer for SGD with momentum, a scheduler that reduces the learning rate when the validation error
             does not decrease and the cross-entropy loss function
    """
    model = YOLOv1(S=S,
                   B=B,
                   C=C,
                   mode='classification').to(DEVICE)

    optimizer = opt.SGD(params=model.parameters(),
                        lr=INIT_LR,
                        momentum=MOMENTUM,
                        weight_decay=WEIGHT_DECAY)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=PATIENCE, min_lr=MIN_LR)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    train_dataset = ImageNet(root=IMAGENET_DIR_PATH,
                             split='train',
                             transform=transforms.Compose([transforms.Resize(RESIZE_D),
                                                           transforms.CenterCrop(RESIZE_D),
                                                           transforms.RandomCrop(INPUT_D),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.ColorJitter(),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225]),
                                                           transforms.RandomErasing()]))

    val_dataset = ImageNet(root=IMAGENET_DIR_PATH,
                           split='val',
                           transform=transforms.Compose([transforms.Resize(RESIZE_D),
                                                         transforms.CenterCrop(INPUT_D),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])]))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=MINI_BATCH,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              shuffle=SHUFFLE,
                              drop_last=DROP_LAST)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=MINI_BATCH,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY)

    return train_loader, val_loader, model, optimizer, scheduler, criterion


def init_train(model: YOLOv1,
               optimizer: opt.SGD,
               scheduler: lr_scheduler.ReduceLROnPlateau) -> Tuple[int, List[float], List[float]]:
    """
    Initialize the epoch, the training and the validation loss history lists, the model, the optimizer and the learning
    rate scheduler.

    :param model: The YOLO (pretraining) classification model
    :param optimizer: The SGD with momentum optimizer that will be used for training
    :param scheduler: The scheduler that will reduce the learning rate on validation loss plateaus
    :return: The current epoch and two lists containing the average train validation losses up to this epoch.
    """
    if TRAIN_MODEL:
        if LOAD_MODEL and os.path.exists(CHECKPOINT_PATH):
            checkpoint = th.load(CHECKPOINT_PATH)

            epoch = checkpoint['epoch']
            train_loss_history = checkpoint['train_loss_history']
            val_loss_history = checkpoint['val_loss_history']

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            epoch = 0
            train_loss_history = []
            val_loss_history = []
    else:
        pretrained_model_weights = th.load(PRETRAINED_MODEL_WEIGHTS)
        model.load_state_dict(pretrained_model_weights)

        epoch, train_loss_history, val_loss_history = None, None, None

    return epoch, train_loss_history, val_loss_history


def main():
    train_loader, val_loader, model, optimizer, scheduler, criterion = setup_train()
    epoch, train_loss_history, val_loss_history = init_train(model, optimizer, scheduler)
    if TRAIN_MODEL:
        train(train_loader, val_loader, model, optimizer, scheduler, criterion, epoch,
              train_loss_history, val_loss_history)
    top1_accuracy, top5_accuracy = measure_accuracy(model, val_loader)

    print(f'Single-Crop Top1 Accuracy = {top1_accuracy * 100:.2f}%')
    print(f'Single-Crop Top5 Accuracy = {top5_accuracy * 100:.2f}%')

    if TRAIN_MODEL:
        print(f'Train Loss History: {train_loss_history}')
        print(f'Validation Loss History: {val_loss_history}')


if __name__ == '__main__':
    main()
