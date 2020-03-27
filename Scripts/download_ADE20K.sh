#!/bin/bash

DATASET_DIR=$1

wget -nc -P $DATASET_DIR http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

unzip -n $DATASET_DIR/ADEChallengeData2016.zip -d $DATASET_DIR

rm $DATASET_DIR/ADEChallengeData2016.zip

mv  -v $DATASET_DIR/ADEChallengeData2016/* $DATASET_DIR/

rm -r $DATASET_DIR/ADEChallengeData2016

echo ========================================================================
echo "Set the path below to \"ROOT:\" in the config file from Config/config_ADE20K.yaml:"
echo -e "\033[32m $DATASET_DIR \033[00m"
echo ========================================================================
