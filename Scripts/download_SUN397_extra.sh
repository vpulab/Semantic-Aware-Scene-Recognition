#!/bin/bash

DATASET_DIR=$1

wget -O $DATASET_DIR/SUN_extra.zip https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EbGDvmje34FMv5_eBARgyvgB9l9XIvrL4GNixUI_3OPk0Q?download=1

unzip -n $DATASET_DIR/SUN_extra.zip -d $DATASET_DIR

rm $DATASET_DIR/SUN_extra.zip

echo ========================================================================
echo "Precomputed semantic segmentation information correctly downloaded."
echo ========================================================================