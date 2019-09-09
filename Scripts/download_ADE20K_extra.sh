#!/bin/bash

DATASET_DIR=$1

wget -O $DATASET_DIR/ADE20K_extra.zip https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EaDcaRv17bFAqZjin_kzXk8BdDPWTj79Uy-ONzBDn2TfvQ?download=1

unzip -n $DATASET_DIR/ADE20K_extra.zip -d $DATASET_DIR

rm $DATASET_DIR/ADE20K_extra.zip

echo ========================================================================
echo "Precomputed semantic segmentation information correctly downloaded."
echo ========================================================================