#!/bin/bash

DATASET_DIR=$1

wget  -O $DATASET_DIR/Places_extra.zip https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/Ee877Dqa0CFMqgJyjvj-HO8BND0yXhNviKSC0LDa8ABptw?download=1

unzip -n $DATASET_DIR/Places_extra.zip -d $DATASET_DIR

rm $DATASET_DIR/Places_extra.zip

echo ========================================================================
echo "Set the path below to \"ROOT:\" in the config file from config/config_Places365.yaml:"
echo -e "\033[32m $DATASET_DIR \033[00m"
echo ========================================================================