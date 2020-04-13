# Histopathologic Cancer Detection model

Convolutional neural network model for Histopathologic Cancer Detection based on a modified version of PatchCamelyon dataset that achives >0.98 AUROC on Kaggle private test set. 


## Reproducing solution
To reproduce my solution without retraining, do the following steps:
1. [Installation](#installation)
2. [Download Dataset](#download-dataset)
3. [Download Pretrained models](#pretrained-models)
4. [Inference](#inference)
5. [Additional details](#details)

## Installation
All requirements should be detailed in requirements.txt. Using virtual environment is strongly recommended.
```
$ python3 -m venv venv_hist
$ source venv_hist/bin/activate
$ pip install -r requirements.txt
```

## Download dataset
Download and extract *train_images.zip* and *test_images.zip* to *data* directory.
```
$ kaggle competitions download -c histopathologic-cancer-detection
$ wget https://storage.googleapis.com/kaggle-forum-message-attachments/496876/11666/patch_id_wsi_full.zip 
$ unzip patch_id_wsi_full.zip -d data
$ unzip histopathologic-cancer-detection.zip -d data
$ rm *.zip
```

### Generate CSV files
```
$ python tools/create_splits.py
```
## Training
In the configs directory, you can find configurations I used to train my final models.

To train models, run following commands.
```
$ python train.py --config_name {config_path} 
```

## Pretrained models
You can download pretrained model that were used for my final model from [link](https://www.kaggle.com/ivanpan/histopathologic-cancer-detection-weights)
```
$ mkdir -p weights
$ bash download_pretrained.sh
```


## Inference
If trained weights are prepared, you can create files that contain predictions for test set using testing config files from configs directory.

In order to inference a single model run:
```
$ python inference.py --config_name configs/{}
```
In order to inference a blend (simple mean) of several models run:
```
$ python inference.py --config_name configs/test/test_blend.yml
```

After that you can find .csv files in subs directory. Keep in mind that test predictions are generated with test time augmentation (TTA-4) by default, which makes inference several times slower. 

## Details 
- Training logs (training and validation metrics) are stored in /.runs directory for tensorboard. 
- Basic logs (training and validation metrics, hyperparameters, scheduling info, etc) are also stored in /logs directory.
- TTA is done using great repo [ttach](https://github.com/qubvel/ttach).
- Best weights (based on AUROC on validation) are stored in /weights directory. You can find SWA code in tools/utils.py. If you decide to use it, just uncomment saving model weights at the end of each epoch. 
