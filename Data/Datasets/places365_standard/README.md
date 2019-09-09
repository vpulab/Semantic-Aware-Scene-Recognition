# Places 365 Dataset Setup
Instructions to download and setup [Places 365 Dataset]([http://places2.csail.mit.edu/index.html](http://places2.csail.mit.edu/index.html)).

## Setup
### RGB Images

 1. To download Places 365 training and validation sets (RGB Images) run the following script (~25 Gb) from the repository folder:
	 
	    bash ./Scripts/download_Places365.sh [PATH]
	   
	   For example:

	    bash ./Scripts/download_Places365.sh ./Data/Datasets/places365_standard
	    
 2. Copy the [PATH] to the `config_Places365.yaml` configuration file:
 
		DATASET:
		    NAME: places365_standard
		    ROOT: [WRITE PATH HERE]

### [RECOMMENDED] Precomputed Semantic Segmentation Information

1. To download precomputed Semantic Segmentation maks for faster training and validation (labels and scores) run the following script (~1GB) from the repository folder:

		bash ./Scripts/download_Places365_extra.sh [PATH FOR DATASET]  

	For example

		bash ./Scripts/download_Places365_extra.sh ./Data/Datasets/places365_standard

2. To use precomputed semantic segmentation activate the option `PRECOMPUTED_SEM` in `config_Places365.yaml` configuration file:
	
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