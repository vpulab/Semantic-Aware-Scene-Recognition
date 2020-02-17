#!/bin/bash

DATASET_DIR=$1

wget -O $DATASET_DIR/ADE20K_extra.zip http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/ADE20K_extra.zip

unzip -n $DATASET_DIR/ADE20K_extra.zip -d $DATASET_DIR

rm $DATASET_DIR/ADE20K_extra.zip

echo ========================================================================
echo "Precomputed semantic segmentation information correctly downloaded."
echo ========================================================================
