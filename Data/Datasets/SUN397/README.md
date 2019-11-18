# SUN 397 Dataset Setup
Proceed with the following instructions to download and setup SUN 397 Dataset [[1](https://groups.csail.mit.edu/vision/SUN/)].

## Setup
### RGB Images

 1. To download SUN 397 training and validation sets (RGB Images) run the following script (~200 MB) from the repository folder:
	 
	    bash ./Scripts/download_SUN397.sh [PATH]
	   
	   For example:

	    bash ./Scripts/download_SUN397.sh ./Data/Datasets/SUN397
	    
 2. Copy the [PATH] to the `config_SUN397.yaml` configuration file:
 
		DATASET:
		    NAME: SUN397
		    ROOT: [WRITE PATH HERE]

### [RECOMMENDED] Precomputed Semantic Segmentation Information
Precomputed Semantic Segmentation includes two folders:
 - noisy_annotations_RGB: This folder includes Top@3 Semantic labels per pixel in a 3-channel image.
 - noisy_scores_RGB: This folder contains Top@3 scores per pixel in a 3-channel image.

In order to download follow the next steps:

1. To download precomputed Semantic Segmentation maks for faster training and validation (labels and scores) run the following script (~1GB) from the repository folder:

		bash ./Scripts/download_SUN397_extra.sh [PATH FOR DATASET]  

	For example

		bash ./Scripts/download_SUN397_extra.sh ./Data/Datasets/SUN397

2. To use precomputed semantic segmentation activate the option `PRECOMPUTED_SEM` in `config_SUN397.yaml` configuration file:
	
		PRECOMPUTED_SEM: TRUE

## Dataset Structure
The dataset must follow the following structure:
```
|----train
	|--- airport_inside
		airport_inside_0001.jpg
		...
	|--- gym
		gym_0001.jpg
		...	
|----val
	|--- airport_inside
		airport_inside_0005.jpg
		...
	|--- gym
		gym_0005.jpg
		...		
|----noisy_annotations_RGB
	|--- train
		|--- airport_inside
			airport_inside_0001.png
			...
		|--- gym
			gym_0001.jpg
			...
	--- val
		|--- airport_inside
			airport_inside_0005.png
			...
		|--- gym
			gym_0005.jpg
			...
|----noisy_scores_RGB
	|--- train
		|--- airport_inside
			airport_inside_0001.png
			...
		|--- gym
			gym_0001.jpg
			...
	--- val
		|--- airport_inside
			airport_inside_0005.png
			...
		|--- gym
			gym_0005.jpg
			...
```

Note that *noisy_annotations* and *noisy_scores* folders are only necessary if precomputed semantic segmentation is used.