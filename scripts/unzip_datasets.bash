SRC_DIR='./datasets/'
TGT_DIR='./datasets/'

unzip $SRC_DIR"test900_ov2.zip" -d $TGT_DIR
unzip $SRC_DIR"test360_ov3.zip" -d $TGT_DIR

DATASETS=("test1800_ov1" "train10000_ov2_1" "train10000_ov2_2" "train20000_ov1_1" "train20000_ov1_2" "train3500_ov3_1" "train3500_ov3_2")

for dataset in ${DATASETS[@]}; do
    zip -s 0 $SRC_DIR$dataset".zip" --out $SRC_DIR$dataset"_agg.zip"
    unzip $SRC_DIR$dataset"_agg.zip" -d $TGT_DIR
    rm $SRC_DIR$dataset"_agg.zip"
done
