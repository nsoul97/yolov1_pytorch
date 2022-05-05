import torch as th
import torchvision.transforms as transforms
import torch.optim as opt
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import VOC_Detection
from transforms import RandomScaleTranslate, Resize, RandomColorJitter, RandomHorizontalFlip, ToYOLOTensor
from model import YOLOv1
from loss import YOLO_Loss
from tqdm import tqdm
from typing import List, Tuple, Dict

# Model Hyperparameters
S = 7
B = 2
D = 448

# Loss Function Hyperparameters
L_COORD = 5.0
L_NOOBJ = 0.5

# Data Augmentation Hyperparameters
HUE = 0.1
SATURATION = 1.5
EXPOSURE = 1.5

RESIZE_PROB = 0.2
ZOOM_OUT_PROB = 0.4
ZOOM_IN_PROB = 0.4
JITTER = 0.2

# Data Loading Hyperparameters
BATCH = 64
SUBDIVISIONS = 8
NUM_WORKERS = 10
SHUFFLE = True
PIN_MEMORY = True
DROP_LAST = True

# Training Hyperparameters
MAX_EPOCHS = 156
INIT_LR = 0.0005
BURN_IN = 100
BURN_IN_POW = 2.
LR_SCHEDULE = [(750, 2.0),  # (step, scale)
               (1500, 2.0),
               (2250, 1.25),
               (3250, 1.60),
               (5500, 1.25),
               (15000, 0.8),
               (20000, 0.625),
               (25000, 0.8),
               (30000, 0.5),
               (35000, 0.5)]
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# VOC Dataset Directory
PASCAL_VOC_DIR_PATH = "/media/soul/DATA/cv_datasets/PASCAL_VOC/VOC_Detection"

# Compute Device (use a GPU if available)
DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

# Checkpoint Hyperparameters
LOAD_MODEL = 'pretrain'  # 'pretrain', 'train', None
PRETRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                           "Detection/checkpoints/pretrained_model_weights.pt"
TRAINING_CHECKPOINT_PATH = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                           "Detection/checkpoints/training_checkpoint.pt"
TRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                        "Detection/checkpoints/trained_model_weights.pt"
CHECKPOINT_T = 10


##########################################################################

class MultiStepScaleLR:
    """
    The MultiStepScaleLR class implements a custom learning rate scheduler. For the first weight updates, the learning
    rate increases to avoid divergence. After these burn-in batches, the learning is multiplied with the specified value
    for each of the corresponding steps.
    """
    def __init__(self,
                 optimizer: opt.SGD,
                 init_lr: float,
                 lr_schedule: List[Tuple[int, float]],
                 burn_in: int,
                 burn_in_pow: float) -> None:
        """
        Initialize the MultiStepScaleLR object. The optimizer, the steps and their corresponding scales and the burn-in
        batches and power are saved. A batch counter that measures the number of times the network's weight have been
        updated is initialized to 0. Furthermore, a variable is initialized to point to the next step during which the
        learning rate will be scaled.

        :param optimizer: The SGD with momentum optimizer
        :param init_lr: The learning rate after burn-in batches
        :param lr_schedule: The steps and their corresponding scales
        :param burn_in: The number of batches during which the learning rate increases in the beginning of the training
        :param burn_in_pow: The burn in power that specifies the rate at which the learning rates increases at the
                             beginning of the training.
        """
        self.optimizer = optimizer
        self.steps, self.scales = zip(*lr_schedule)
        self.burn_in = burn_in
        self.init_lr = init_lr
        self.pow = burn_in_pow
        self.batch = 0
        self.next_step_ind = 0

    def step(self) -> None:
        """
        Update the learning rate of the optimizer. During the first burn in batches, the learning rate increases to
        reach the given init_lr. Afterwards the learning rate is scaled as specified at the corresponding steps.
        """
        self.batch += 1
        if self.batch < self.burn_in:
            self.optimizer.param_groups[0]['lr'] = self.init_lr * ((self.batch+1)/self.burn_in)**self.pow
        elif self.next_step_ind < len(self.steps) and self.batch == self.steps[self.next_step_ind]:
            self.optimizer.param_groups[0]['lr'] *= self.scales[self.next_step_ind]
            self.next_step_ind += 1

    def state_dict(self) -> dict:
        """
        The function returns the state dictionary of the object that can be loaded to resume training when
        LOAD_MODEL='train'.

        :return: A dictionary with the members of the class as keys and their corresponding values. The optimizer is not
                 saved in the dictionary.
        """
        return {key: value for (key, value) in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the given state dictionary to resume training when LOAD_MODEL='train'.

        :param state_dict: A dictionary with the members of the class as keys and their corresponding values.
                           The optimizer is not saved in the dictionary.
        """
        self.__dict__.update(state_dict)


def train_epoch(train_loader: DataLoader,
                model: YOLOv1,
                optimizer: opt.SGD,
                criterion: YOLO_Loss,
                scheduler: MultiStepScaleLR,
                mini_batch: int) -> Tuple[float, int]:
    """
    Train the YOLO model for one epoch and return the average loss per training instance for this epoch.

    :param train_loader: The DataLoader of the PASCAL VOC train set
    :param model: The YOLOv1 detection model
    :param optimizer: The SGD with momentum optimizer
    :param criterion: The YOLO loss criterion
    :param scheduler: The learning rate scheduler
    :param mini_batch: The mini batch counter
    :return: The average training loss per instance for this epoch and the updated mini_batch counter
    """
    av_loss = 0.

    model.train()
    for x, y_gt in train_loader:
        mini_batch += 1
        x, y_gt = x.to(DEVICE), y_gt.to(DEVICE)
        y_pred = model(x)
        loss = criterion(y_pred, y_gt) / SUBDIVISIONS
        loss.backward()

        if mini_batch == SUBDIVISIONS:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            mini_batch = 0

        av_loss += loss.item() * SUBDIVISIONS

    av_loss /= len(train_loader)
    return av_loss, mini_batch


def validate_epoch(val_loader: DataLoader,
                   model: YOLOv1,
                   criterion: YOLO_Loss) -> float:
    """
    Validate the YOLO model and return the average loss per test instance for this epoch.

    :param val_loader: The DataLoader of the PASCAL VOC test set
    :param model: The YOLOv1 detection model
    :param criterion: The YOLO loss criterion
    :return: The average test loss per instance for this epoch
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
          test_loader: DataLoader,
          model: YOLOv1,
          optimizer: opt.SGD,
          criterion: YOLO_Loss,
          scheduler: MultiStepScaleLR,
          epoch: int,
          mini_batch: int,
          train_loss_history: List[float],
          test_loss_history: List[float]) -> None:
    """
    Train the YOLOv1 model for a MAX_EPOCHS number of epochs. Plot a bar to illustrate the progress of training.
    Every CHECKPOINT_T epochs and after the training has finished, create a new checkpoint.

    :param train_loader: The DataLoader of the PASCAL VOC train set
    :param test_loader: The DataLoader of the PASCAL VOC test set
    :param model: The YOLOv1 detection model
    :param optimizer: The SGD with momentum optimizer
    :param criterion: The YOLO loss criterion
    :param scheduler: The learning rate scheduler, which scales the learning at the specified step with the given factor
    :param epoch: The starting epoch of the training
    :param mini_batch: The mini batch counter [0, SUBDIVISIONS]
    :param train_loss_history: The history of the training losses up to the current epoch
    :param test_loss_history: The history of the test losses up to the current epoch
    """

    pbar = tqdm(total=MAX_EPOCHS, desc='Training Epoch', initial=epoch, unit='epoch', position=0, leave=True)
    if mini_batch == 0:
        optimizer.zero_grad()

    while epoch < MAX_EPOCHS:
        epoch += 1

        train_loss, mini_batch = train_epoch(train_loader, model, optimizer, criterion, scheduler, mini_batch)
        test_loss = validate_epoch(test_loader, model, criterion)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        if epoch % CHECKPOINT_T == 0:
            th.save({'epoch': epoch,
                     'mini_batch': mini_batch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(),
                     'train_loss_history': train_loss_history,
                     'test_loss_history': test_loss_history,
                     'grads': {p[0]: p[1].grad for p in model.named_parameters()}
                     }, TRAINING_CHECKPOINT_PATH)

        pbar.set_postfix_str(f'Train Loss={train_loss:.3f}, Test Loss={test_loss:.3f}')
        pbar.update(1)

    th.save(model.state_dict(), TRAINED_MODEL_WEIGHTS)
    pbar.close()


def setup_train():
    """
    Instantiate the model, the optimizer, the learning rate scheduler, the loss criterion, the train and the test
    PASCAL VOC datasets and their corresponding loaders.

    :return: The DataLoader objects of the training and the test PASCAL VOC dataset, the YOLO detection model,
             an SGD with momentum optimizer, a learning rate scheduler that scales the learning at the given steps by
             the corresponding factors, and the YOLO loss criterion.
    """
    model = YOLOv1(S=S,
                   B=B,
                   C=VOC_Detection.C).to(DEVICE)

    optimizer = opt.SGD(params=model.parameters(),
                        lr=INIT_LR * (1/BURN_IN)**BURN_IN_POW,
                        momentum=MOMENTUM,
                        weight_decay=WEIGHT_DECAY)

    scheduler = MultiStepScaleLR(optimizer,
                                 init_lr=INIT_LR,
                                 lr_schedule=LR_SCHEDULE,
                                 burn_in=BURN_IN,
                                 burn_in_pow=BURN_IN_POW)

    criterion = YOLO_Loss(S=S,
                          C=VOC_Detection.C,
                          B=B,
                          D=D,
                          L_coord=L_COORD,
                          L_noobj=L_NOOBJ).to(DEVICE)

    train_dataset = VOC_Detection(root_dir=PASCAL_VOC_DIR_PATH,
                                  split='train',
                                  transforms=transforms.Compose([
                                      RandomScaleTranslate(output_size=D,
                                                           jitter=JITTER,
                                                           resize_p=RESIZE_PROB,
                                                           zoom_out_p=ZOOM_OUT_PROB,
                                                           zoom_in_p=ZOOM_IN_PROB),
                                      RandomColorJitter(hue=HUE,
                                                        sat=SATURATION,
                                                        exp=EXPOSURE),
                                      RandomHorizontalFlip(p=0.5),
                                      ToYOLOTensor(S=S,
                                                   C=VOC_Detection.C,
                                                   normalize=[[0.4549, 0.4341, 0.4010],
                                                              [0.2703, 0.2672, 0.2808]])]))

    test_dataset = VOC_Detection(root_dir=PASCAL_VOC_DIR_PATH,
                                 split='test',
                                 transforms=transforms.Compose([
                                     Resize(output_size=D),
                                     ToYOLOTensor(S=S,
                                                  C=VOC_Detection.C,
                                                  normalize=[[0.4549, 0.4341, 0.4010],
                                                             [0.2703, 0.2672, 0.2808]])]))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH // SUBDIVISIONS,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              shuffle=SHUFFLE,
                              drop_last=DROP_LAST)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH // SUBDIVISIONS,
                             num_workers=NUM_WORKERS,
                             pin_memory=PIN_MEMORY)

    return train_loader, test_loader, model, optimizer, scheduler, criterion


def init_train(model: YOLOv1,
               optimizer: opt.SGD,
               scheduler: MultiStepScaleLR) -> Tuple[int, int, List[float], List[float]]:
    """
    Initialize the epoch, the training and the validation loss history lists, the model, the optimizer and the learning
    rate scheduler.

    :param model: The YOLO (pretraining) classification model
    :param optimizer: The SGD with momentum optimizer that will be used for training
    :param scheduler: The scheduler that will reduce the learning rate on validation loss plateaus
    :return: The current epoch and mini-batch counter and two lists containing the average train and test losses up
             to this epoch.
    """
    if LOAD_MODEL is None:
        epoch = 0
        mini_batch = 0
        train_loss_history = []
        test_loss_history = []

    elif LOAD_MODEL == 'pretrain':
        pretrained_model_weights = th.load(PRETRAINED_MODEL_WEIGHTS)
        model.load_state_dict(pretrained_model_weights, strict=False)
        epoch = 0
        mini_batch = 0
        train_loss_history = []
        test_loss_history = []

    elif LOAD_MODEL == 'train':
        checkpoint = th.load(TRAINING_CHECKPOINT_PATH)

        epoch = checkpoint['epoch']
        mini_batch = checkpoint['mini_batch']
        train_loss_history = checkpoint['train_loss_history']
        test_loss_history = checkpoint['test_loss_history']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        for p in model.named_parameters():
            p[1].grad = checkpoint['grads'][p[0]]

    else:
        assert 0

    return epoch, mini_batch, train_loss_history, test_loss_history


def main():
    train_loader, test_loader, model, optimizer, scheduler, criterion = setup_train()
    epoch, mini_batch, train_loss_hist, test_loss_hist = init_train(model, optimizer, scheduler)
    train(train_loader, test_loader, model, optimizer, criterion, scheduler,
          epoch, mini_batch,
          train_loss_hist, test_loss_hist)


if __name__ == '__main__':
    main()
