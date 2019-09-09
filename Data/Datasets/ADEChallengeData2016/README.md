# ADE20K Dataset Setup
Instructions to download and setup [ADE20K Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/).

## Setup
### RGB Images

 1. To download ADE20K training and validation sets (RGB Images) run the following script (~1GB) from the repository folder:
	 
	    bash ./Scripts/download_ADE20K.sh [PATH]
	   
	   For example:

	    bash ./Scripts/download_ADE20K.sh ./Data/Datasets/ADEChallengeData2016
	    
 2. Copy the [PATH] to the `config_ADE20K.yaml` configuration file:
 
		DATASET:
	    NAME: ADEChallengeData2016
	    ROOT: [WRITE PATH HERE]

### [RECOMMENDED] Precomputed Semantic Segmentation Information

1. To download precomputed Semantic Segmentation maks for faster training and validation (labels and scores) run the following script (~3GB) from the repository folder:

		bash ./Scripts/download_ADE20K_extra.sh [PATH FOR DATASET]  

	For example

		bash ./Scripts/download_ADE20K_extra.sh ./Data/Datasets/ADEChallengeData2016

2. To use precomputed semantic segmentation activate the option `PRECOMPUTED_SEM` in `config_ADE20K.yaml` configuration file:
	
		PRECOMPUTED_SEM: TRUE

## Dataset Structure
The dataset must follow the following structure:
```
|----images
	|--- training
		ADE_train_00000001.jpg
		...
	|--- validation
		ADE_val_00000001.jpg
		...
|----noisy_annotations_RGB
	|--- training
		ADE_train_00000001.png
		...
	|--- validation
		ADE_train_00000001.png
		...
|----noisy_scores_RGB
	|--- training
		ADE_train_00000001.png
		...
	|--- validation
		ADE_train_00000001.png
		...
```

Note that *noisy_annotations* and *noisy_scores* folders are only necessary if precomputed semantic segmentation is used.