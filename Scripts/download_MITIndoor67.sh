#!/bin/bash

DATASET_DIR=$1

wget  -O $DATASET_DIR/MIT.zip http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/MITIndoor67.zip

unzip -n $DATASET_DIR/MIT.zip -d $DATASET_DIR

rm $DATASET_DIR/MIT.zip

echo ========================================================================
echo "Set the path below to \"ROOT:\" in the config file from Config/config_MITIndoor.yaml:"
echo -e "\033[32m $DATASET_DIR \033[00m"
echo ========================================================================
