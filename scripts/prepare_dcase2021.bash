
TGT_DIR='datasets/DCASE2021'

pip3 install zenodo_get
zenodo_get 10.5281/zenodo.5476980 -o $TGT_DIR
zip -s 0 $TGT_DIR/foa_dev.zip --out $TGT_DIR/foa_dev_agg.zip
zip -s 0 $TGT_DIR/mic_dev.zip --out $TGT_DIR/mic_dev_agg.zip
unzip $TGT_DIR/foa_dev_agg.zip -d $TGT_DIR
unzip $TGT_DIR/mic_dev_agg.zip -d $TGT_DIR
unzip $TGT_DIR/metadata_dev.zip -d $TGT_DIR 
unzip $TGT_DIR/foa_eval.zip -d $TGT_DIR
unzip $TGT_DIR/mic_eval.zip -d $TGT_DIR
unzip $TGT_DIR/metadata_eval.zip -d $TGT_DIR
rm $TGT_DIR/*.z*

mv $TGT_DIR/foa_dev/*/*.wav $TGT_DIR/foa_dev
mv $TGT_DIR/mic_dev/*/*.wav $TGT_DIR/mic_dev
mv $TGT_DIR/metadata_dev/*/*.csv $TGT_DIR/metadata_dev
mv $TGT_DIR/foa_eval/*/*.wav $TGT_DIR/foa_eval
mv $TGT_DIR/mic_eval/*/*.wav $TGT_DIR/mic_eval
mv $TGT_DIR/metadata_eval/*/*.csv $TGT_DIR/metadata_eval
rm -r $TGT_DIR/foa_dev/dev-*
rm -r $TGT_DIR/mic_dev/dev-* 
rm -r $TGT_DIR/metadata_dev/dev-*
rm -r $TGT_DIR/foa_eval/eval-*
rm -r $TGT_DIR/mic_eval/eval-*
rm -r $TGT_DIR/metadata_eval/eval-*

python src/preproc.py dataset=DCASE2021 wav_format=.wav
python src/preproc.py dataset=DCASE2021 dataset_type=eval wav_format=.wav

