# Histopathologic Cancer Detection model
Convolutional neural network model for Histopathologic Cancer Detection based on a modified version of PatchCamelyon dataset that achives 

## Reproducing solution
To reproduce my solution without retraining, do the following steps:
1. [Installation](#installation)
2. [Download Dataset](#download-dataset)
3. [Download Pretrained models](#pretrained-models)
4. run `bash reproduce.sh`

## Installation
All requirements should be detailed in requirements.txt. Using virtual environment is strongly recommended.
```
python3 -m venv venv_hist
source venv_hist/source/bin/activate
pip install -r requirements.txt
```

## Prepare dataset
### Download dataset
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
