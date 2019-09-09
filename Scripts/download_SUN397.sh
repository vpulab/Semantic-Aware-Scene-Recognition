#!/bin/bash

DATASET_DIR=$1

wget  -O $DATASET_DIR/SUN.zip https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EWz4fZwqQsVFrsM7D8Q0qdcB3JuHGfLj7_k9G7NViMWfqQ?download=1

unzip -n $DATASET_DIR/SUN.zip -d $DATASET_DIR

rm $DATASET_DIR/SUN.zip

echo ========================================================================
echo "Set the path below to \"ROOT:\" in the config file from config/config_SUN397.yaml:"
echo -e "\033[32m $DATASET_DIR \033[00m"
echo ========================================================================