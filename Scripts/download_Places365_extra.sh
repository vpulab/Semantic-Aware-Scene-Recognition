#!/bin/bash

DATASET_DIR=$1

wget  -O $DATASET_DIR/Places_extra.zip http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/places365_standard_extra_val.zip

unzip -n $DATASET_DIR/Places_extra.zip -d $DATASET_DIR

rm $DATASET_DIR/Places_extra.zip

echo ========================================================================
echo "Set the path below to \"ROOT:\" in the config file from config/config_Places365.yaml:"
echo -e "\033[32m $DATASET_DIR \033[00m"
echo ========================================================================
