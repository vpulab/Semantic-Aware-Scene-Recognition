#!/bin/bash

DATASET_DIR=$1

wget  -O $DATASET_DIR/SUN.zip http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/SUN397.zip

unzip -n $DATASET_DIR/SUN.zip -d $DATASET_DIR

rm $DATASET_DIR/SUN.zip

echo ========================================================================
echo "Set the path below to \"ROOT:\" in the config file from Config/config_SUN397.yaml:"
echo -e "\033[32m $DATASET_DIR \033[00m"
echo ========================================================================
