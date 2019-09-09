from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import random
import torchvision.transforms.functional as TF
import numpy as np
import torch
from imgaug import augmenters as iaa


class ADE20KDataset(Dataset):
    """Class for ADE20K dataset."""

    def __init__(self, root_dir, set, tencrops=False, clean=False, SemRGB=True):
        """
        Initialize the dataset
        :param root_dir: Root directory to the dataset
        :param set: Dataset set: Training or Validation
        :param clean: Use the cleaned version of ADE20K instead of the original one
        """
        # Extract main path and set (Train or Val)
        self.image_dir = root_dir
        self.set = set
        # Set boolean variable of ten crops for validation
        self.TenCrop = tencrops

        if clean:
            self.clean = "_clean"
        else:
            self.clean = ""

        self.SemRGB = SemRGB
        if SemRGB:
            self.RGB = "_RGB"
        else:
            self.RGB = ""

        # Decode dataset scene categories
        self.classes = list()
        class_file_name = os.path.join(root_dir, "Scene_Names" + self.clean + ".txt")

        with open(class_file_name) as class_file:
            for line in class_file:
                self.classes.append(line.split()[0])

        self.nclasses = self.classes.__len__()

        # Create list for filenames and scene ground-truth labels
        self.filenames = list()
        self.labels = list()
        self.labelsindex = list()
        filenames_file = os.path.join(root_dir, ("sceneCategories_" + set + self.clean + ".txt"))

        with open(filenames_file) as class_file:
            for line in class_file:
                name, label = line.split()
                self.filenames.append(name)
                self.labels.append(label)
                self.labelsindex.append(self.classes.index(label))

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
        self.resizeSize = 256
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
                transforms.Resize(self.resizeSize),
                transforms.CenterCrop(self.outputSize),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.STD)
            ])

            if not SemRGB:
                self.val_transforms_sem = transforms.Compose([
                    transforms.Resize(self.resizeSize, interpolation=Image.NEAREST),
                    transforms.CenterCrop(self.outputSize),
                    transforms.Lambda(lambda sem: torch.unsqueeze(torch.from_numpy(np.asarray(sem) + 1).long(), 0))
                ])

                self.val_transforms_scores = transforms.Compose([
                    transforms.Resize(self.resizeSize),
                    transforms.CenterCrop(self.outputSize),
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda semScore: torch.unsqueeze(torch.from_numpy(np.asarray(semScore) + 1).long(), 0)),
                ])
            else:
                self.val_transforms_sem = transforms.Compose([
                    transforms.Resize(self.resizeSize, interpolation=Image.NEAREST),
                    transforms.CenterCrop(self.outputSize),
                    transforms.Lambda(lambda sem: torch.from_numpy(np.asarray(sem) + 1).long().permute(2, 0, 1))
                ])

                self.val_transforms_scores = transforms.Compose([
                    transforms.Resize(self.resizeSize),
                    transforms.CenterCrop(self.outputSize),
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda semScore: torch.from_numpy(np.asarray(semScore)).long().permute(2, 0, 1)),
                ])

        else:
            self.val_transforms_img = transforms.Compose([
                transforms.Resize(self.resizeSize),
                transforms.TenCrop(self.outputSize),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(self.mean, self.STD)(crop) for crop in crops])),
            ])

            if not SemRGB:
                self.val_transforms_sem = transforms.Compose([
                    transforms.Resize(self.resizeSize, interpolation=Image.NEAREST),
                    transforms.TenCrop(self.outputSize),
                    transforms.Lambda(lambda crops: torch.stack([torch.unsqueeze(torch.from_numpy(np.asarray(crop) + 1).long(), 0) for crop in crops]))
                ])

                self.val_transforms_scores = transforms.Compose([
                    transforms.Resize(self.resizeSize),
                    transforms.TenCrop(self.outputSize),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    # transforms.Lambda(lambda crops: torch.stack([torch.unsqueeze(torch.from_numpy(np.asarray(crop) + 1).long(), 0) for crop in crops])),
                ])
            else:
                self.val_transforms_sem = transforms.Compose([
                    transforms.Resize(self.resizeSize, interpolation=Image.NEAREST),
                    transforms.TenCrop(self.outputSize),
                    transforms.Lambda(lambda crops: torch.stack([torch.from_numpy(np.asarray(crop) + 1).long().permute(2, 0, 1) for crop in crops])),
                ])

                self.val_transforms_scores = transforms.Compose([
                    transforms.Resize(self.resizeSize),
                    transforms.TenCrop(self.outputSize),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    # transforms.Lambda(lambda crops: torch.stack([torch.from_numpy(np.asarray(crop) + 1).long().permute(2, 0, 1) for crop in crops])),
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
        img_name = os.path.join(self.image_dir, "images", self.set, (self.filenames[idx] + ".jpg"))
        img = Image.open(img_name)

        # Convert it to RGB if gray-scale
        if img.mode is not "RGB":
            img = img.convert("RGB")

        # Load semantic segmentation ground-truth
        # semGT_name = os.path.join(self.image_dir, "annotations", self.set, (self.filenames[idx] + ".png"))
        # semGT = Image.open(semGT_name)

        # Load semantic segmentation mask
        sem_name = os.path.join(self.image_dir, ("noisy_annotations" + self.RGB), self.set, (self.filenames[idx] + ".png"))
        sem = Image.open(sem_name)

        # Load semantic segmentation scores
        sem_score_name = os.path.join(self.image_dir, ("noisy_scores" + self.RGB), self.set, (self.filenames[idx] + ".png"))
        semScore = Image.open(sem_score_name)

        # Apply transformations depending on the set (train, val)
        if self.set is "training":
            # Define Random crop. If image is smaller resize first.
            bilinearResize_trans = transforms.Resize(self.resizeSize)
            nearestResize_trans = transforms.Resize(self.resizeSize, interpolation=Image.NEAREST)

            img = bilinearResize_trans(img)
            # semGT = nearestResize_trans(semGT)
            sem = nearestResize_trans(sem)
            semScore = bilinearResize_trans(semScore)

            # Extract Random Crop parameters
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.outputSize, self.outputSize))
            # Apply Random Crop parameters
            img = TF.crop(img, i, j, h, w)
            # semGT = TF.crop(semGT, i, j, h, w)
            sem = TF.crop(sem, i, j, h, w)
            semScore = TF.crop(semScore, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                img = TF.hflip(img)
                # semGT = TF.hflip(semGT)
                sem = TF.hflip(sem)
                semScore = TF.hflip(semScore)

            # Apply transformations from ImgAug library
            img = np.asarray(img)
            # semGT = np.asarray(semGT)
            sem = np.asarray(sem)
            semScore = np.asarray(semScore)

            img = np.squeeze(self.seq.augment_images(np.expand_dims(img, axis=0)))
            # semGT = np.squeeze(self.seq_sem.augment_images(np.expand_dims(np.expand_dims(semGT, 0), 3)))
            if self.SemRGB:
                sem = np.squeeze(self.seq_sem.augment_images(np.expand_dims(sem, 0)))
                semScore = np.squeeze(self.seq_sem.augment_images(np.expand_dims(semScore, 0)))
            else:
                sem = np.squeeze(self.seq_sem.augment_images(np.expand_dims(np.expand_dims(sem, 0), 3)))
                semScore = np.squeeze(self.seq_sem.augment_images(np.expand_dims(np.expand_dims(semScore, 0), 3)))

            # Apply not random transforms. To tensor and normalization for RGB. To tensor for semantic segmentation.
            img = self.train_transforms_img(img)
            # semGT = self.train_transforms_sem(semGT)
            sem = self.train_transforms_sem(sem)
            semScore = self.train_transforms_scores(semScore)
        else:
            img = self.val_transforms_img(img)
            # semGT = self.val_transforms_sem(semGT)
            sem = self.val_transforms_sem(sem)
            semScore = self.val_transforms_scores(semScore)

        if not self.TenCrop:
            if not self.SemRGB:
                assert img.shape[0] == 3 and img.shape[1] == self.outputSize and img.shape[2] == self.outputSize
                # assert semGT.shape[0] == 1 and semGT.shape[1] == self.outputSize and semGT.shape[2] == self.outputSize
                assert sem.shape[0] == 1 and sem.shape[1] == self.outputSize and sem.shape[2] == self.outputSize
                assert semScore.shape[0] == 1 and semScore.shape[1] == self.outputSize and semScore.shape[2] == self.outputSize
            else:
                assert img.shape[0] == 3 and img.shape[1] == self.outputSize and img.shape[2] == self.outputSize
                assert sem.shape[0] == 3 and sem.shape[1] == self.outputSize and sem.shape[2] == self.outputSize
                assert semScore.shape[0] == 3 and semScore.shape[1] == self.outputSize and semScore.shape[2] == self.outputSize
        else:
            if not self.SemRGB:
                assert img.shape[0] == 10 and img.shape[2] == self.outputSize and img.shape[3] == self.outputSize
                # assert semGT.shape[0] == 10 and semGT.shape[2] == self.outputSize and semGT.shape[3] == self.outputSize
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
