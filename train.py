import numpy as np
import os
import argparse
import yaml
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import shutil

import utils

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default='configs/train.yml')

    parser_args = parser.parse_args()

    with open(vars(parser_args)['config_name'], 'r') as config_file:
        hyperparams = yaml.full_load(config_file)

    return hyperparams


if __name__ == "__main__":
    hyperparams = parse_config()

    fold_index = hyperparams['train']['fold_index']
    backbone = hyperparams['train']['backbone']
    n_epochs = hyperparams['train']['n_epochs']
    data_dir = hyperparams['dataset']['data_dir']
    optimizer_params_first_stage = hyperparams['optimizer']['params_first_stage']
    optimizer_params_second_stage = hyperparams['optimizer']['params_second_stage']
    scheduler_params = hyperparams['scheduler']
    loss_params = hyperparams['loss']

    batch_sizes = {'train_batch_size': hyperparams['dataloaders']['train_batch_size'],
                   'valid_batch_size': hyperparams['dataloaders']['valid_batch_size']}
    transforms = utils.build_augmentations('train')

    loaders = utils.build_dataloaders(data_dir, transforms, 'train', batch_sizes, fold_index=fold_index)
    model = utils.build_model(backbone).cuda()
    model.unfreeze_model()

    optim = utils.build_optim(model, optimizer_params_first_stage, scheduler_params, loss_params)
    criterion, optimizer, scheduler = optim['criterion'], optim['optimizer'], optim['scheduler']

    shutil.rmtree('runs/hist_model_{}_fold_{}'.format(backbone, fold_index), ignore_errors=True)
    writer = SummaryWriter('runs/hist_model_{}_fold_{}'.format(backbone, fold_index))

    valid_loss_min = np.Inf
    for epoch in (range(n_epochs)):
        print(time.ctime(), 'Epoch:', epoch)

        train_metrics = utils.train_epoch(epoch, model, optimizer, criterion, loaders['train_loader'])
        validation_metrics = utils.validation(epoch, model, criterion, loaders['valid_loader'])

        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/validation', validation_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/validation', validation_metrics['accuracy'], epoch)
        writer.add_scalar('AUC/validation', validation_metrics['val_auc'], epoch)

        print('Epoch {}, train loss: {:.4f}, valid loss: {:.4f}'.format(epoch, train_metrics['loss'], validation_metrics['loss']))
        if epoch >= 2:
            scheduler.step(validation_metrics['loss'])

        if epoch == 1:
            model.unfreeze_model()
            optimizer = utils.build_optim(model, optimizer_params_second_stage, scheduler_params, loss_params)['optimizer']

        if validation_metrics['loss'] <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                validation_metrics['loss']))
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), 'weights/model_{}_fold{}.pth'.format(backbone, fold_index))
            valid_loss_min = validation_metrics['loss']

    writer.close()
