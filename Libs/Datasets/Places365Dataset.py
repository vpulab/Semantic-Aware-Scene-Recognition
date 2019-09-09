from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import random
import torchvision.transforms.functional as TF
import numpy as np
import torch
from imgaug import augmenters as iaa


class Places365Dataset(Dataset):
    """Class for Places 365 dataset."""

    def __init__(self, root_dir, set, tencrops=False, SemRGB=False):
        """
        Initialize the dataset. Read scene categories, get number of classes, create filename and ground-truth labels
        lists, create ImAug and PyTorch transformations

        :param root_dir: Root directory to the dataset
        :param set: Dataset set: Training or Validation
        """
        # Extract main path and set (Train or Val).
        self.image_dir = root_dir
        self.set = set
        # Set boolean variable of ten crops for validation
        self.TenCrop = tencrops

        self.SemRGB = SemRGB
        if SemRGB:
            self.RGB = "_RGB"
        else:
            self.RGB = ""

        # Decode dataset scene categories
        self.classes = list()
        class_file_name = os.path.join(root_dir, "categories_places365.txt")

        with open(class_file_name) as class_file:
            for line in class_file:
                line = line.split()[0]
                split_indices = [i for i, letter in enumerate(line) if letter == '/']
                # Check if there a class with a subclass inside (outdoor, indoor)
                if len(split_indices) > 2:
                    line = line[:split_indices[2]] + '-' + line[split_indices[2]+1:]

                self.classes.append(line[split_indices[1] + 1:])

        # Get number of classes
        self.nclasses = self.classes.__len__()

        # Create list for filenames and scene ground-truth labels
        self.filenames = list()
        self.labels = list()
        self.labelsindex = list()
        self.auxiliarnames = list()
        filenames_file = os.path.join(root_dir, (set + ".txt"))

        # Fill filenames list and ground-truth labels list
        with open(filenames_file) as class_file:
            for line in class_file:
                # if random.random() > 0.6 or (self.set is "val"):
                split_indices = [i for i, letter in enumerate(line) if letter == '/']
                # Obtain name and label
                name = line[split_indices[1] + 1:-1]
                label = line[split_indices[0] + 1: split_indices[1]]

                self.filenames.append(name)
                self.labels.append(label)
                self.labelsindex.append(self.classes.index(label))
                str2 = "\n"
                indx2 = line.find(str2)
                self.auxiliarnames.append(line[0:indx2])

        # Control Statements for data loading
        assert len(self.filenames) == len(self.labels)

        # ----------------------------- #
        #     ImAug Transformations     #
        # ----------------------------- #
        # Transformations for train set
        self.seq = iaa.Sequential([
            # Small gaussian blur with random sigma between 0 and 0.5.
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # Add gaussian noise.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
        ], random_order=True)  # apply augmenters in random order

        self.seq_sem = iaa.Sequential([
            iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
        ], random_order=True)

        # ----------------------------- #
        #    Pytorch Transformations    #
        # ----------------------------- #
        self.mean = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.outputSize = 224

        # Train Set Transformation
        self.train_transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.STD)
        ])
        self.train_transforms_scores = transforms.ToTensor()

        if not SemRGB:
            self.train_transforms_sem = transforms.Lambda(
                lambda sem: torch.unsqueeze(torch.from_numpy(np.asarray(sem) + 1).long(), 0))
        else:
            self.train_transforms_sem = transforms.Lambda(
                lambda sem: torch.from_numpy(np.asarray(sem) + 1).long().permute(2, 0, 1))

        # Transformations for validation set
        if not self.TenCrop:
            self.val_transforms_img = transforms.Compose([
                transforms.CenterCrop(self.outputSize),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.STD)
            ])

            if not SemRGB:
                self.val_transforms_sem = transforms.Compose([
                    transforms.CenterCrop(self.outputSize),
                    transforms.Lambda(lambda sem: torch.unsqueeze(torch.from_numpy(np.asarray(sem) + 1).long(), 0))
                ])

                self.val_transforms_scores = transforms.Compose([
                    transforms.CenterCrop(self.outputSize),
                    transforms.ToTensor(),
                ])
            else:
                self.val_transforms_sem = transforms.Compose([
                    transforms.CenterCrop(self.outputSize),
                    transforms.Lambda(lambda sem: torch.from_numpy(np.asarray(sem) + 1).long().permute(2, 0, 1))
                ])

                self.val_transforms_scores = transforms.Compose([
                    transforms.CenterCrop(self.outputSize),
                    transforms.ToTensor(),
                ])

        else:
            self.val_transforms_img = transforms.Compose([
                transforms.TenCrop(self.outputSize),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(
                    lambda crops: torch.stack([transforms.Normalize(self.mean, self.STD)(crop) for crop in crops])),
            ])

            if not SemRGB:
                self.val_transforms_sem = transforms.Compose([
                    transforms.TenCrop(self.outputSize),
                    transforms.Lambda(lambda crops: torch.stack(
                        [torch.unsqueeze(torch.from_numpy(np.asarray(crop) + 1).long(), 0) for crop in crops]))
                ])

                self.val_transforms_scores = transforms.Compose([
                    transforms.TenCrop(self.outputSize),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                ])
            else:
                self.val_transforms_sem = transforms.Compose([
                    transforms.TenCrop(self.outputSize),
                    transforms.Lambda(lambda crops: torch.stack(
                        [torch.from_numpy(np.asarray(crop) + 1).long().permute(2, 0, 1) for crop in crops])),
                ])

                self.val_transforms_scores = transforms.Compose([
                    transforms.TenCrop(self.outputSize),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                ])

    def __len__(self):
        """
        Function to get the size of the dataset
        :return: Size of dataset
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Function to get a sample from the dataset. First both RGB and Semantic images are read in PIL format. Then
        transformations are applied from PIL to Numpy arrays to Tensors.

        For regular usage:
            - Images should be outputed with dimensions (3, W, H)
            - Semantic Images should be outputed with dimensions (1, W, H)

        In the case that 10-crops are used:
            - Images should be outputed with dimensions (10, 3, W, H)
            - Semantic Images should be outputed with dimensions (10, 1, W, H)

        :param idx: Index
        :return: Dictionary containing {RGB image, semantic segmentation mask, scene category index}
        """

        # Get RGB image path and load it
        # img_name = os.path.join(self.image_dir, self.set, self.labels[idx], self.filenames[idx])
        img_name = os.path.join(os.path.join(self.image_dir, self.auxiliarnames[idx]))
        img = Image.open(img_name)

        # Convert it to RGB if gray-scale
        if img.mode is not "RGB":
            img = img.convert("RGB")

        filename_sem = img_name[img_name.find('places365_standard') + 19:img_name.find('.jpg')]
        # aux_indx = img_name.find('train')
        if img_name.find('/train/') > 0:
            sem_name = os.path.join('/media/vpu/f376732d-7565-499a-95f5-b6b26c4a902d/Datasets/places365_standard',
                                    "noisy_annotations_RGB", (filename_sem + ".png"))
            sem_score_name = os.path.join('/media/vpu/f376732d-7565-499a-95f5-b6b26c4a902d/Datasets/places365_standard',
                                          "noisy_scores_RGB", (filename_sem + ".png"))
        else:
            sem_name = os.path.join(self.image_dir, "noisy_annotations_RGB", (filename_sem + ".png"))
            sem_score_name = os.path.join(self.image_dir, "noisy_scores_RGB", (filename_sem + ".png"))

        sem = Image.open(sem_name)
        semScore = Image.open(sem_score_name)

        # Load semantic segmentation mask
        # if self.set == 'train':
        #     filename_sem = self.filenames[idx][0:self.filenames[idx].find('.jpg')]
        #     sem_name = os.path.join('/media/vpu/f376732d-7565-499a-95f5-b6b26c4a902d/Datasets/places365_standard',
        #                             "noisy_annotations_RGB", self.set, self.labels[idx], (filename_sem + ".png"))
        #
        #     sem = Image.open(sem_name)
        #
        #     # Load semantic segmentation scores
        #     filename_scores = self.filenames[idx][0:self.filenames[idx].find('.jpg')]
        #     sem_score_name = os.path.join('/media/vpu/f376732d-7565-499a-95f5-b6b26c4a902d/Datasets/places365_standard',
        #                                   "noisy_scores_RGB", self.set, self.labels[idx], (filename_scores + ".png"))
        #
        #     semScore = Image.open(sem_score_name)
        #
        # else:
        #     filename_sem = self.filenames[idx][0:self.filenames[idx].find('.jpg')]
        #     sem_name = os.path.join(self.image_dir, "noisy_annotations_RGB", self.set, self.labels[idx], (filename_sem + ".png"))
        #
        #     sem = Image.open(sem_name)
        #
        #     # Load semantic segmentation scores
        #     filename_scores = self.filenames[idx][0:self.filenames[idx].find('.jpg')]
        #     sem_score_name = os.path.join(self.image_dir, "noisy_scores_RGB", self.set, self.labels[idx], (filename_scores + ".png"))
        #
        #     semScore = Image.open(sem_score_name)

        # Apply transformations depending on the set (train, val)
        if self.set is "train":
            # # Extract Random Crop parameters
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.outputSize, self.outputSize))
            # Apply Random Crop parameters
            img = TF.crop(img, i, j, h, w)
            sem = TF.crop(sem, i, j, h, w)
            semScore = TF.crop(semScore, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                img = TF.hflip(img)
                sem = TF.hflip(sem)
                semScore = TF.hflip(semScore)

            # Apply transformations from ImgAug library
            img = np.asarray(img)
            sem = np.asarray(sem)
            semScore = np.asarray(semScore)

            img = np.squeeze(self.seq.augment_images(np.expand_dims(img, axis=0)))
            if self.SemRGB:
                sem = np.squeeze(self.seq_sem.augment_images(np.expand_dims(sem, 0)))
                semScore = np.squeeze(self.seq_sem.augment_images(np.expand_dims(semScore, 0)))
            else:
                sem = np.squeeze(self.seq_sem.augment_images(np.expand_dims(np.expand_dims(sem, 0), 3)))
                semScore = np.squeeze(self.seq_sem.augment_images(np.expand_dims(np.expand_dims(semScore, 0), 3)))

            # Apply not random transforms. To tensor and normalization for RGB. To tensor for semantic segmentation.
            img = self.train_transforms_img(img)
            sem = self.train_transforms_sem(sem)
            semScore = self.train_transforms_scores(semScore)
        else:
            img = self.val_transforms_img(img)
            sem = self.val_transforms_sem(sem)
            semScore = self.val_transforms_scores(semScore)

        # Final control statements
        if not self.TenCrop:
            if not self.SemRGB:
                assert img.shape[0] == 3 and img.shape[1] == self.outputSize and img.shape[2] == self.outputSize
                assert sem.shape[0] == 1 and sem.shape[1] == self.outputSize and sem.shape[2] == self.outputSize
                assert semScore.shape[0] == 1 and semScore.shape[1] == self.outputSize and semScore.shape[2] == self.outputSize
            else:
                assert img.shape[0] == 3 and img.shape[1] == self.outputSize and img.shape[2] == self.outputSize
                assert sem.shape[0] == 3 and sem.shape[1] == self.outputSize and sem.shape[2] == self.outputSize
                assert semScore.shape[0] == 3 and semScore.shape[1] == self.outputSize and semScore.shape[2] == self.outputSize
        else:
            if not self.SemRGB:
                assert img.shape[0] == 10 and img.shape[2] == self.outputSize and img.shape[3] == self.outputSize
                assert sem.shape[0] == 10 and sem.shape[2] == self.outputSize and sem.shape[3] == self.outputSize
                assert semScore.shape[0] == 10 and semScore.shape[2] == self.outputSize and semScore.shape[3] == self.outputSize
            else:
                assert img.shape[0] == 10 and img.shape[2] == self.outputSize and img.shape[3] == self.outputSize
                assert sem.shape[0] == 10 and sem.shape[2] == self.outputSize and sem.shape[3] == self.outputSize
                assert semScore.shape[0] == 10 and semScore.shape[2] == self.outputSize and semScore.shape[3] == self.outputSize

        # Create dictionary
        self.sample = {'Image': img, 'Semantic': sem,
                       'Semantic Scores': semScore, 'Scene Index': self.classes.index(self.labels[idx])}

        return self.sample
