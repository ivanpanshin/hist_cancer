import pandas as pd
import os
import yaml
import argparse
from sklearn.model_selection import GroupKFold

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_name", type=str, default='configs/train/train_efficientnet-b3-fold0.yml')
    parser_args = parser.parse_args()

    with open(vars(parser_args)['config_name'], 'r') as config_file:
        hyperparams = yaml.full_load(config_file)['dataset']

    return hyperparams

if __name__ == "__main__":
    hyperparams = parse_config()
    data_dir = hyperparams['data_dir']
    n_splits = hyperparams['n_splits']

    os.makedirs('splits', exist_ok=True)

    train_labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
    wsi = pd.read_csv(os.path.join(data_dir, 'patch_id_wsi_full.csv'))

    wsi_ids = wsi.wsi.values

    group_kfold = GroupKFold(n_splits=n_splits)

    for fold_index, (train_index, valid_index) in enumerate(group_kfold.split(train_labels.id, train_labels.label, wsi_ids)):

        X_train, X_valid = train_labels.id[train_index], train_labels.id[valid_index]
        y_train, y_valid = train_labels.label[train_index], train_labels.label[valid_index]

        X_train.to_csv('splits/X_train_fold_{}.csv'.format(fold_index), index=False)
        X_valid.to_csv('splits/X_valid_fold_{}.csv'.format(fold_index), index=False)

        y_train.to_csv('splits/y_train_fold_{}.csv'.format(fold_index), index=False)
        y_valid.to_csv('splits/y_valid_fold_{}.csv'.format(fold_index), index=False)
