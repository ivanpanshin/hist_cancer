kaggle datasets download ivanpan/histopathologic-cancer-detection-weights -f model_efficientnet-b3_fold0.pth -p weights
kaggle datasets download ivanpan/histopathologic-cancer-detection-weights -f model_efficientnet-b3_fold1.pth -p weights
kaggle datasets download ivanpan/histopathologic-cancer-detection-weights -f model_efficientnet-b3_fold2.pth -p weights
kaggle datasets download ivanpan/histopathologic-cancer-detection-weights -f model_efficientnet-b3_fold3.pth -p weights
kaggle datasets download ivanpan/histopathologic-cancer-detection-weights -f model_efficientnet-b3_fold4.pth -p weights

kaggle datasets download ivanpan/histopathologic-cancer-detection-weights -f model_seresnet50_fold0.pth -p weights
kaggle datasets download ivanpan/histopathologic-cancer-detection-weights -f model_seresnet50_fold1.pth -p weights
kaggle datasets download ivanpan/histopathologic-cancer-detection-weights -f model_seresnet50_fold2.pth -p weights
kaggle datasets download ivanpan/histopathologic-cancer-detection-weights -f model_seresnet50_fold3.pth -p weights
kaggle datasets download ivanpan/histopathologic-cancer-detection-weights -f model_seresnet50_fold4.pth -p weights

unzip weights/model_efficientnet-b3_fold0.pth.zip -d weights
unzip weights/model_efficientnet-b3_fold1.pth.zip -d weights
unzip weights/model_efficientnet-b3_fold2.pth.zip -d weights
unzip weights/model_efficientnet-b3_fold3.pth.zip -d weights
unzip weights/model_efficientnet-b3_fold4.pth.zip -d weights

unzip weights/model_seresnet50_fold0.pth.zip -d weights
unzip weights/model_seresnet50_fold1.pth.zip -d weights
unzip weights/model_seresnet50_fold2.pth.zip -d weights
unzip weights/model_seresnet50_fold3.pth.zip -d weights
unzip weights/model_seresnet50_fold4.pth.zip -d weights

rm weights/*.zip
