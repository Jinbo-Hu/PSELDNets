
# optional: install kaggle
DOWNLOAD_PATH='datasets/L3DAS22'
kaggle datasets download -d l3dasteam/l3das22 -p $DOWNLOAD_PATH --force --unzip
mv $DOWNLOAD_PATH/L3DAS22_Task2_train/L3DAS22_Task2_train/data/* $DOWNLOAD_PATH/data_train
mv $DOWNLOAD_PATH/L3DAS22_Task2_dev/L3DAS22_Task2_dev/data/* $DOWNLOAD_PATH/data_train
mv $DOWNLOAD_PATH/L3DAS22_Task2_test/L3DAS22_Task2_test/data/* $DOWNLOAD_PATH/data_test
mv $DOWNLOAD_PATH/L3DAS22_Task2_train/L3DAS22_Task2_train/labels/* $DOWNLOAD_PATH/labels_train
mv $DOWNLOAD_PATH/L3DAS22_Task2_dev/L3DAS22_Task2_dev/labels/* $DOWNLOAD_PATH/labels_train
mv $DOWNLOAD_PATH/L3DAS22_Task2_test/L3DAS22_Task2_test/labels/* $DOWNLOAD_PATH/labels_test
rm -r $DOWNLOAD_PATH/L3DAS22_Task2*
rm -r $DOWNLOAD_PATH/L3DAS22_Task1*

python src/preproc.py dataset=L3DAS22 wav_format=.wav data=l3das22/default
python src/preproc.py dataset=L3DAS22 dataset_type=eval wav_format=.wav data=l3das22/default
