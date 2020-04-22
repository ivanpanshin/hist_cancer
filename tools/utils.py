import numpy as np
import pandas as pd
import os
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim import lr_scheduler
import cv2
import albumentations
from albumentations import pytorch as AT
import random
import ttach as tta
import PIL.Image as Image

from . import seresnet


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def add_to_logs(logging, message):
    logging.info(message)


def add_to_tensorboard_logs(writer, message, tag, index):
    writer.add_scalar(tag, message, index)


def swa(paths):
    state_dicts = []
    for path in paths:
        state_dicts.append(torch.load(path))

    average_dict = OrderedDict()
    for k in state_dicts[0].keys():
        average_dict[k] = sum([state_dict[k] for state_dict in state_dicts]) / len(state_dicts)

    return average_dict


class cancer_dataset(Dataset):
    def __init__(self, data_dir, mode, transform, ids=None, labels=None):
        self.data_dir = data_dir
        self.mode = mode
        self.ids = ids
        self.transform = transform

        if self.mode == 'train':
            self.labels = labels

        if self.mode == 'test':
            self.ids = [x.split('.')[0] for x in os.listdir(os.path.join(self.data_dir, 'test'))]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_name = os.path.join(self.data_dir, self.mode, self.ids[idx] + '.tif')
        # img = cv2.imread(image_name)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image = self.transform(image=img)
        # image = image['image']

        image = Image.open(image_name)
        image = self.transform(image)

        if self.mode == 'train':
            return image, self.labels[idx]
        return image


def train_epoch(epoch, model, optimizer, criterion, train_loader):
    model.train()
    train_loss = []

    for batch_i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[:, 1], target.float())
        train_loss.append(loss.item())

        # a = target.data.cpu().numpy()
        # b = output[:,-1].detach().cpu().numpy()

        loss.backward()
        optimizer.step()

        del data, target, output
        torch.cuda.empty_cache()

    train_loss = np.mean(train_loss)

    metrics = {'loss': train_loss}

    return metrics


def validation(epoch, model, criterion, valid_loader):
    model.eval()
    model = tta.ClassificationTTAWrapper(model, tta.aliases.d4_transform())
    val_loss = []

    correct_samples = 0

    full_a = np.zeros(len(valid_loader) * valid_loader.batch_size)
    full_b = np.zeros(len(valid_loader) * valid_loader.batch_size)

    for batch_i, (data, target) in enumerate(valid_loader):
        if batch_i % 100 == 0:
            print(batch_i * valid_loader.batch_size)
        with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output[:, 1], target.float())

            val_loss.append(loss.item())

            full_a[batch_i * valid_loader.batch_size: batch_i * valid_loader.batch_size + output.shape[0]] = target.data.cpu().numpy()
            full_b[batch_i * valid_loader.batch_size: batch_i * valid_loader.batch_size + output.shape[0]] = output[:, -1].detach().cpu().numpy()

            correct_samples += (target.detach().cpu().numpy() == np.argmax(output.detach().cpu().numpy(), axis=1)).sum()

            del data, target, output
            torch.cuda.empty_cache()

    valid_loss = np.mean(val_loss)
    val_auc = roc_auc_score(full_a, full_b)
    accuracy_score = correct_samples / (len(valid_loader) * valid_loader.batch_size)

    metrics = {'loss': valid_loss, 'accuracy': accuracy_score, 'val_auc': val_auc}
    print(metrics)

    return metrics


def test_preds(model, data_dir, test_loader):
    model.eval()

    model = tta.ClassificationTTAWrapper(model, tta.aliases.d4_transform())

    sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    test_preds = np.zeros(sample_submission.shape[0])

    for batch_i, data in enumerate(test_loader):
        with torch.no_grad():
            data = data.cuda()
            output = model(data)

            test_preds[batch_i * test_loader.batch_size: batch_i * test_loader.batch_size + output.shape[0]] = sigmoid(output[:, 1].detach().cpu().numpy())

            del data, output
            torch.cuda.empty_cache()
        if batch_i % 100 == 0:
            print(batch_i * test_loader.batch_size)

    return test_preds


class Model(nn.Module):
    def __init__(self, backbone):
        super(Model, self).__init__()
        self.net = EfficientNet.from_pretrained(backbone)

        num_features = self.net._bn1.num_features * 2
        for param in self.net.parameters():
            param.requires_grad = False
        self.net._fc = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=0.8),
            nn.Linear(num_features, num_features // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features // 2),
            nn.Dropout(p=0.8),
            nn.Linear(num_features // 2, num_features // 4, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features // 4),
            nn.Dropout(p=0.8),
            nn.Linear(num_features // 4, 2))

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

    def unfreeze_model(self):
        for param in self.net.parameters():
            param.requires_grad = True

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        x = self.net.extract_features(x)

        x1 = self.avg_pooling(x)
        x2 = self.max_pooling(x)
        x = torch.cat([x1, x2], 1)
        x = x.view(x.size(0), -1)
        x = self.net._dropout(x)
        x = self.net._fc(x)

        return x


def build_model(backbone):
    if backbone == 'seresnet50':
        return seresnet.se_re()
    elif backbone.split('-')[0] == 'efficientnet':
        return Model(backbone)


def build_augmentations(mode):
    if mode == 'train':
        train_transforms = transforms.Compose([
            transforms.Resize((196, 196)),
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(contrast=0.5),
                transforms.ColorJitter(saturation=0.5),
                transforms.ColorJitter(hue=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            ]),
            transforms.RandomChoice([
                transforms.RandomRotation((0, 0)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270)),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.RandomRotation((90, 90)),
                ]),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.RandomRotation((270, 270)),
                ])
            ]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

        valid_transforms = transforms.Compose([
            transforms.Resize((196, 196)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

        transforms_dict = {'train_transforms': train_transforms, 'valid_transforms': valid_transforms}

    else:

        test_transforms = transforms.Compose([
            transforms.Resize((196, 196)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

        transforms_dict = {'test_transforms': test_transforms}

    return transforms_dict


def build_optim(model, optimizer_params, scheduler_params, loss_params):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), **optimizer_params)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params['params'])

    return {'criterion': criterion, 'optimizer': optimizer, 'scheduler': scheduler}


def build_dataloaders(data_dir, transforms, mode, batch_sizes, fold_index=None):
    if mode == 'train':
        X_train = pd.read_csv('splits/X_train_fold_{}.csv'.format(fold_index))
        X_valid = pd.read_csv('splits/X_valid_fold_{}.csv'.format(fold_index))

        y_train = pd.read_csv('splits/y_train_fold_{}.csv'.format(fold_index))
        y_valid = pd.read_csv('splits/y_valid_fold_{}.csv'.format(fold_index))

        train_dataset = cancer_dataset(data_dir, 'train', transforms['train_transforms'], X_train.id.values, y_train.label.values)
        valid_dataset = cancer_dataset(data_dir, 'train', transforms['valid_transforms'], X_valid.id.values, y_valid.label.values)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sizes['train_batch_size'])
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_sizes['valid_batch_size'])

        loaders = {'train_loader': train_loader, 'valid_loader': valid_loader}
    else:
        test_set = cancer_dataset(data_dir, 'test', transforms['test_transforms'])
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_sizes['test_batch_size'])

        loaders = {'test_loader': test_loader}

    return loaders
