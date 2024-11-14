
TGT_DIR='datasets/STARSS23'

pip3 install zenodo_get
zenodo_get 10.5281/zenodo.7880637 -d $TGT_DIR
unzip $TGT_DIR/foa_dev.zip -d $TGT_DIR
unzip $TGT_DIR/mic_dev.zip -d $TGT_DIR
unzip $TGT_DIR/metadata_dev.zip -d $TGT_DIR
unzip $TGT_DIR/foa_eval.zip -d $TGT_DIR
unzip $TGT_DIR/mic_eval.zip -d $TGT_DIR
mv $TGT_DIR/foa_dev/*/*.wav $TGT_DIR/foa_dev
mv $TGT_DIR/mic_dev/*/*.wav $TGT_DIR/mic_dev
mv $TGT_DIR/metadata_dev/*/*.csv $TGT_DIR/metadata_dev
rm -r $TGT_DIR/foa_dev/dev-*
rm -r $TGT_DIR/mic_dev/dev-* 
rm -r $TGT_DIR/metadata_dev/dev-*

python src/preproc.py dataset=STARSS23 wav_format=.wav