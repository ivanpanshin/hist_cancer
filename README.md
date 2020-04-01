# Histopathologic Cancer Detection model
Convolutional neural network model for Histopathologic Cancer Detection based on a modified version of PatchCamelyon dataset that achives 

## Reproducing solution
To reproduce my solution without retraining, do the following steps:
1. [Installation](#installation)
2. [Download Dataset](#download-dataset)
3. [Download Pretrained models](#pretrained-models)
4. [Inference](#inference)

## Installation
All requirements should be detailed in requirements.txt. Using virtual environment is strongly recommended.
```
python3 -m venv venv_hist
source venv_hist/source/bin/activate
pip install -r requirements.txt
```

## Download dataset
Download and extract *train_images.zip* and *test_images.zip* to *data* directory.
```
$ kaggle competitions download -c histopathologic-cancer-detection
$ wget https://storage.googleapis.com/kaggle-forum-message-attachments/496876/11666/patch_id_wsi_full.zip 
$ unip patch_id_wsi_full.zip -d data
$ unzip histopathologic.zip -d data
$ chmod 644 data/*
```

### Generate CSV files
```
$ python create_splits.py
```
## Training
In the configs directory, you can find configurations I used to train my final models.

To train models, run following commands.
```
$ python train.py --config_name {config_path} 
```

## Pretrained models
You can download pretrained model that were used for my final model from [link](https://www.kaggle.com/pudae81/understandingclouds1stplaceweights)
```
$ mkdir -p weights
$ bash download_pretrained.sh
```


## Inference
If trained weights are prepared, you can create files that contain predictions for test set using testing config files from configs directory.

In order to inference a single model run:
```
$ python inference.py --config_name configs/test_single_model.yml
```
In order to inference a blend (simple mean) of several models run:
```
$ python inference.py --config_name configs/test_blend.yml
```

After that you can find .csv files in subs directory. Keep in mind that test predictions are generated with test time augmentation (TTA-4) by default, which makes inference several times slower. 


