kaggle datasets download ivanpan/histopathologic-cancer-detection-weights -f model_efficientnet-b0_fold0.pth -p weights
kaggle datasets download ivanpan/histopathologic-cancer-detection-weights -f model_efficientnet-b1_fold0.pth -p weights
unzip weights/model_efficientnet-b0_fold0.pth.zip -d weights
unzip weights/model_efficientnet-b1_fold0.pth.zip -d weights
rm weights/*.zip
