#!/bin/bash

DATASET_DIR=$1

wget -O $DATASET_DIR/MIT_extra.zip https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/ERPnW_w_o7pGi9B-TQoM048Bcwoi2Aggx44t40de7-bytw?download=1

unzip -n $DATASET_DIR/MIT_extra.zip -d $DATASET_DIR

rm $DATASET_DIR/MIT_extra.zip

echo ========================================================================
echo "Precomputed semantic segmentation information correctly downloaded."
echo ========================================================================