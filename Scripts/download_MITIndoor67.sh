#!/bin/bash

DATASET_DIR=$1

wget  -O $DATASET_DIR/MIT.zip https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EdXZlW6RidhIphjkefdF4oEBwx6BYjT2y7oZn9HojJcSJQ?download=1

unzip -n $DATASET_DIR/MIT.zip -d $DATASET_DIR

rm $DATASET_DIR/MIT.zip

echo ========================================================================
echo "Set the path below to \"ROOT:\" in the config file from config/config_MITIndoor.yaml:"
echo -e "\033[32m $DATASET_DIR \033[00m"
echo ========================================================================