import yaml
import os
import numpy as np
import pandas as pd
import argparse
import time
from tools import utils
import logging


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default='configs/test/test_efficientnet-b3-fold0.yml')

    parser_args = parser.parse_args()

    with open(vars(parser_args)['config_name'], 'r') as config_file:
        hyperparams = yaml.full_load(config_file)

    return hyperparams


if __name__ == "__main__":
    start_time = time.ctime()
    hyperparams = parse_config()
    single_model = hyperparams['blend']['single_model']
    data_dir = hyperparams['dataset']['data_dir']
    batch_sizes = {'test_batch_size': hyperparams['dataloaders']['test_batch_size']}

    transforms = utils.build_augmentations('test')
    loaders = utils.build_dataloaders(data_dir, transforms, 'test', batch_sizes)

    list_of_model_weights = os.listdir('weights')
    sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    test_preds = np.zeros(sample_submission.shape[0])
    if single_model:
        backbone = hyperparams['blend']['backbone']
        fold = hyperparams['blend']['fold']
        model = utils.build_model(backbone).cuda()
        path = os.path.join('weights', 'model_' + backbone + '_fold' + str(fold) + '.pth')
        model.load_model(path)
        test_preds += utils.test_preds(model, 'data', loaders['test_loader'])

    else:
        for index_of_model, num_of_models in enumerate(list_of_model_weights):
            utils.add_to_logs(logging, 'Inferencing model number {}'.format(index_of_model))
            path = os.path.join('weights', list_of_model_weights[index_of_model])
            backbone = path.split('_')[1]
            model = utils.Model(backbone).cuda()
            model.load_model(path)
            test_preds += utils.test_preds(model, 'data', loaders['test_loader'])

        test_preds /= len(list_of_model_weights)
        test_preds = sigmoid(test_preds)

    sample_submission.label = test_preds
    sample_submission.id = loaders['test_loader'].dataset.ids

    os.makedirs('subs', exist_ok=True)
    if single_model:
        sample_submission.to_csv('subs/sub_{}_fold_{}.csv'.format(backbone, fold), index=False)
    else:
        sample_submission.to_csv('subs/sub_blend.csv', index=False)

