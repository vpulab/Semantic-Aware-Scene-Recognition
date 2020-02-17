#!/bin/bash

DATASET_DIR=$1

wget -O $DATASET_DIR/SUN_extra.zip http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/SUN397_extra.zip

unzip -n $DATASET_DIR/SUN_extra.zip -d $DATASET_DIR

rm $DATASET_DIR/SUN_extra.zip

echo ========================================================================
echo "Precomputed semantic segmentation information correctly downloaded."
echo ========================================================================
