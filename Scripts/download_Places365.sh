#!/bin/bash

DATASET_DIR=$1

#wget -nc -P $DATASET_DIR http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar

unzip -n $DATASET_DIR/places365standard_easyformat.tar -d $DATASET_DIR

#rm $DATASET_DIR/places365standard_easyformat.tar

echo ========================================================================
echo "Set the path below to \"ROOT:\" in the config file from config/config_Places365.yaml:"
echo -e "\033[32m $DATASET_DIR \033[00m"
echo ========================================================================