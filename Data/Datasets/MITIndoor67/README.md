# MIT Indoor 67 Dataset Setup
Proceed with the following instructions to download and setup MIT Indoor 67 Dataset [[1](http://web.mit.edu/torralba/www/indoor.html)].

## Setup
### RGB Images

 1. To download MIT Indoor 67  training and validation sets (RGB Images) run the following script (~200 MB) from the repository folder:
	 
	    bash ./Scripts/download_MITIndoor67.sh [PATH]
	   
	   For example:

	    bash ./Scripts/download_MITIndoor67.sh ./Data/Datasets/MITIndoor67
	    
 2. Copy the [PATH] to the `config_MITIndoor.yaml` configuration file:
 
		DATASET:
		    NAME: MITIndoor67
		    ROOT: [WRITE PATH HERE]

### [RECOMMENDED] Precomputed Semantic Segmentation Information
Precomputed Semantic Segmentation includes two folders:
 - noisy_annotations_RGB: This folder includes Top@3 Semantic labels per pixel in a 3-channel image.
 - noisy_scores_RGB: This folder contains Top@3 scores per pixel in a 3-channel image.

In order to download follow the next steps:

1. To download precomputed Semantic Segmentation maks for faster training and validation (labels and scores) run the following script (~1GB) from the repository folder:

		bash ./Scripts/download_MITIndoor67_extra.sh [PATH FOR DATASET]  

	For example

		bash ./Scripts/download_MITIndoor67_extra.sh ./Data/Datasets/MITIndoor67

2. To use precomputed semantic segmentation activate the option `PRECOMPUTED_SEM` in `config_MITIndoor.yaml` configuration file:
	
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
