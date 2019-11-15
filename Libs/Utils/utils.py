import numpy as np
import torch
import shutil
import matplotlib.pyplot as plt
import cv2


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def unNormalizeImage(image, mean=[0.485, 0.456, 0.406], STD=[0.229, 0.224, 0.225]):
    """
    Unnormalizes a numpy array given mean and STD
    :param image: Image to unormalize
    :param mean: Mean
    :param STD: Standard Deviation
    :return: Unnormalize image
    """
    for i in range(0, image.shape[0]):
        image[i, :, :] = (image[i, :, :] * STD[i]) + mean[i]
    return image


def plotTensorImage(image, label="GT Label"):
    """
    Function to plot a PyTorch Tensor image
    :param image: Image to display in Tensor format
    :param mean: Mean of the normalization
    :param STD: Standard Deviation of the normalization
    :param label: (Optional) Ground-truth label
    :return:
    """
    # Convert PyTorch Tensor to Numpy array
    npimg = image.numpy()
    # # Unnormalize image
    unNormalizeImage(npimg)
    # Change from (chns, rows, cols) to (rows, cols, chns)
    npimg = np.transpose(npimg, (1, 2, 0))

    # Convert to RGB if gray-scale
    if npimg.shape[2] is 1:
        rgbArray = np.zeros((npimg.shape[0], npimg.shape[1], 3), 'float32')
        rgbArray[:, :, 0] = npimg[:, :, 0]
        rgbArray[:, :, 1] = npimg[:, :, 0]
        rgbArray[:, :, 2] = npimg[:, :, 0]
        npimg = rgbArray

    # Display image
    plt.figure()
    plt.imshow(npimg)
    plt.title(label)


class AverageMeter(object):
    """
    Class to store instant values, accumulated and average of measures
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum2 = 0
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sum2 += np.power(val,2) * n
        self.count += n
        self.avg = self.sum / self.count
        self.std = np.sqrt((self.sum2 / self.count) - np.power(self.avg, 2))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Saves check point
    :param state: Dictionary to save. Constains models state_dictionary
    :param is_best: Boolean variable to check if is the best model
    :param filename: Saving filename
    :return:
    """
    torch.save(state, 'Files/' + filename + '_latest.pth.tar')
    if is_best:
        print('Best model updated.')
        shutil.copyfile('Files/' + filename + '_latest.pth.tar', 'Files/' + filename + '_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """
    Computes the top-k accuracy between output and target.
    :param output: output vector from the network
    :param target: ground-truth
    :param topk: Top-k results desired, i.e. top1, top5, top10
    :return: vector with accuracy values
    """
    maxk = max(topk)
    batch_size = target.size(0)
    # output = output.long()
    # target = target.long()

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def getclassAccuracy(output, target, nclasses, topk=(1,)):
    """
    Computes the top-k accuracy between output and target and aggregates it by class
    :param output: output vector from the network
    :param target: ground-truth
    :param nclasses: nclasses in the problem
    :param topk: Top-k results desired, i.e. top1, top2, top5
    :return: topk vectors aggregated by class
    """
    maxk = max(topk)

    score, label_index = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    correct = label_index.eq(torch.unsqueeze(target, 1))

    ClassAccuracyRes = []
    for k in topk:
        ClassAccuracy = torch.zeros([1, nclasses], dtype=torch.uint8).cuda()
        correct_k = correct[:, :k].sum(1)
        for n in range(target.shape[0]):
            ClassAccuracy[0, target[n]] += correct_k[n].byte()
        ClassAccuracyRes.append(ClassAccuracy)

    return ClassAccuracyRes


def scoreRatioMetric(output, target):
    cols = torch.tensor(np.arange(target.size()[0])).long().cuda()
    new_indices = target + cols * output.size()[1]

    GTScores = torch.index_select(output.view(-1), 0, new_indices)

    score, label_index = output.topk(k=1, dim=1, largest=True, sorted=True)

    SR = GTScores / score.squeeze()

    return torch.mean(SR)


def MeanPixelAccuracy(pred, label):
    """
    Function to compute the mean pixel accuracy for semantic segmentation between mini-batch tensors
    :param pred: Tensor of predictions
    :param label: Tensor of ground-truth
    :return: Mean pixel accuracy for all the mini-bath
    """
    # Convert tensors to numpy arrays
    imPred = np.asarray(torch.squeeze(pred))
    imLab = np.asarray(torch.squeeze(label))

    # Create empty numpy arrays
    pixel_accuracy = np.empty(imLab.shape[0])
    pixel_correct = np.empty(imLab.shape[0])
    pixel_labeled = np.empty(imLab.shape[0])

    # Compute pixel accuracy for each pair of images in the batch
    for i in range(imLab.shape[0]):
        pixel_accuracy[i], pixel_correct[i], pixel_labeled[i] = pixelAccuracy(imPred[i], imLab[i])

    # Compute the final accuracy for the batch
    acc = 100.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))

    return acc


def pixelAccuracy(imPred, imLab):
    """
    Computes pixel accuracy between two semantic segmentation images
    :param imPred: Prediction
    :param imLab: Ground-truth
    :return: pixel accuracy
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = torch.sum(imLab > 0).float()
    pixel_correct = torch.sum((imPred == imLab) * (imLab > 0)).float()
    pixel_accuracy = pixel_correct / (pixel_labeled + 1e-10)

    return pixel_accuracy, pixel_correct, pixel_labeled


def semanticIoU(pred, label):
    """
    Computes the mean Intersection over Union for all the classes between two mini-batch tensors of semantic
    segmentation
    :param pred: Tensor of predictions
    :param label: Tensor of ground-truth
    :return: Mean semantic intersection over Union for all the classes
    """
    imPred = np.asarray(torch.squeeze(pred))
    imLab = np.asarray(torch.squeeze(label))

    area_intersection = []
    area_union = []

    for i in range(imLab.shape[0]):
        intersection, union = intersectionAndUnion(imPred[i], imLab[i])
        area_intersection.append(intersection)
        area_union.append(union)

    IoU = 1.0 * np.sum(area_intersection, axis=0) / np.sum(np.spacing(1)+area_union, axis=0)

    return np.mean(IoU)


def intersectionAndUnion(imPred, imLab, numClass=150):
    """
    Computes the intersection and Union for all the classes between two images
    :param imPred: Predictions image
    :param imLab: Ground-truth image
    :param numClass: Number of semantic classes. Default:150
    :return: Intersection and union for all the classes
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0).long()

    # Compute area intersection:
    intersection = imPred * (imPred == imLab).long()
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    IoU = area_intersection / (area_union + 1e-10)

    return IoU


def classSemanticAcc(sceneLabelGT, imPred, imLab, nclasses):

    ClassIoU = torch.zeros([1, nclasses], dtype=torch.float).cuda()

    # Extract batch size
    batch_size = sceneLabelGT.size()[0]

    for n in range(batch_size):
        acc, _, _ = pixelAccuracy(imPred[n, :, :, :]+1, imLab[n, :, :, :])

        ClassIoU[0, sceneLabelGT[n]] = acc

    return ClassIoU


def make_one_hot(labels, semantic_scores, C=151):
    '''
    Converts an integer label torch image to a one-hot tensor of probabilities.

    Parameters
    ----------
    labels : torch.cuda.LongTensor N x H x W / N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification label.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.cuda.FloatTensor N x C x H x W, where C is class number. One-hot encoded.
    '''

    if len(labels.shape) is 3:
        labels = torch.unsqueeze(labels, 1)

    # Semantic Labels
    one_hot = torch.cuda.FloatTensor(labels.size(0), C + 1, labels.size(2), labels.size(3)).zero_()

    # target = one_hot.scatter_(1, labels, semantic_scores.float())
    # target = one_hot.scatter_(1, labels, 1)
    for n in range(labels.size(1)):
        one_hot = one_hot.scatter_(1, torch.unsqueeze(labels[:, n, :, :], 1),
                                   torch.unsqueeze(semantic_scores.float()[:, n, :, :], 1))

    return one_hot


def getHistogramOfClasses(dataloader, classes, set='', ePrint=True):
    """
    Computes the histogram of classes for the given dataloader
    :param dataloader: Pytorch dataloader to compute the histogram
    :param classes: Classes names
    :param set: Indicates the set. Training or validation
    :param ePrint: Enables the information printing
    :return: Histogram of classes
    """
    ClassesHist = [0] * len(classes)
    images = dataloader.dataset.labelsindex
    for item in images:
        ClassesHist[item] += 1
    ClassesHist = np.asarray(ClassesHist)
    N = float(sum(ClassesHist))

    # Compute histogram and percentages
    HistClasses = (ClassesHist / sum(ClassesHist)) * 100
    sortedPercen = np.sort(HistClasses)
    sortedClasses = np.argsort(HistClasses)

    plt.figure()
    # plt.hist(HistClasses, bins=len(classes))
    plt.plot(ClassesHist)
    plt.title(set + " Classes Histogram")

    if ePrint:
        print('{} classes with the most number of samples:'.format(set))
        for c in range(-1, -7, -1):
            print('Class {} with {Percentage:.4f}%'
                  .format(classes[sortedClasses[c]], Percentage=sortedPercen[c]))

        print('{} classes with the less number of samples:'.format(set))
        for c in range(0, 6):
            print('Class {} with {Percentage:.4f}%'
                  .format(classes[sortedClasses[c]], Percentage=sortedPercen[c]))
        print('')

    return ClassesHist


def obtainPredictedClasses(outputSceneLabel):
    """
    Fucntion to obtain the indices for the 10 most-scored scene labels
    :param outputSceneLabel: Tensor obtain from the network
    :return: numpy array 1x10 with scene labels indices
    """
    # Obtain the predicted class by obtaining the maximum score.
    _, pred = outputSceneLabel.topk(10, 1, largest=True, sorted=True)
    idx = pred.cpu().numpy()[0]

    return idx


def saveActivationMap(model, feature_conv, idx, sceneLabelGT, RGB_image, classes, i, set, save=False):
    """
    Computes and saves the Activation Map (AM) obtain by the network
    :param model: Used network and model
    :param feature_conv: Feature map from the last convolutional layer, before the Average Pooling
    :param outputSceneLabel: Predicted scene label
    :param sceneLabelGT: Ground-truth scene label
    :param RGB_image: Input RGB image used to obtain the prediction
    :param classes: List with the scene class names
    :param i: Index to save the image
    :param save: Boolean variable to enable saving
    :return: Rendered activation map and RGB image
    """
    # Obtain the weigths from the last FC layer
    params = list(model.parameters())
    weight_softmax = params[-2].cpu().numpy()
    weight_softmax[weight_softmax < 0] = 0

    # Obtain the Activation Map
    activationMap = returnCAM(feature_conv, weight_softmax, idx[0:3])

    # Render the activation map and the RGB image
    img = RGB_image.squeeze().cpu().numpy()
    img = unNormalizeImage(img)
    img = np.transpose(img, (1, 2, 0)) * 255
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(activationMap[0], (width, height)), cv2.COLORMAP_JET)
    result = cv2.resize(heatmap * 0.4 + img * 0.5, (512, 512))

    # If saving is enabled, save the file to a jpg image within a path
    if save:
        cv2.putText(result, ("GT Label: " + classes[sceneLabelGT.cpu().item()]),
                    org=(20, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=[255, 255, 255])
        cv2.putText(result, "Top1: " + classes[idx[0]],
                    org=(20, 60), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=[255, 255, 255])
        cv2.putText(result, "Top2: " + classes[idx[1]],
                    org=(20, 90), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=[255, 255, 255])
        cv2.putText(result, "Top3: " + classes[idx[2]],
                    org=(20, 120), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=[255, 255, 255])
        Path = '/home/vpu/Semantic-Aware-Scene-Recognition/Results/Scene Activation Maps/' + set + '/'
        cv2.imwrite(Path + 'AM' + str(i+1).zfill(5) + '.jpg', result)

        # Save plain activation map
        Path = '/home/vpu/Multi-Task-Scene-Recognition/Images/Scene Activation Maps Plain/' + set + '/'
        cv2.imwrite(Path + 'AM' + str(i + 1).zfill(5) + '.jpg', heatmap)

    return


def returnCAM(feature_conv, weight_softmax, class_idx):
    """
    Function to compute the Activation Map (AM)
    :param feature_conv: Feature tensor obtain from the last convolutional layer, before the Global Averaging Pooling
    :param weight_softmax: Weigths from the Fully Connected layer that predicts the scene
    :param class_idx: Class index from which the AM is obtained
    :return:
    """
    # Define the upsampling size. ResNet-18 regularly uses (224, 224)
    size_upsample = (224, 224)

    # Obtain sizes from the features
    bz, nc, h, w = feature_conv.shape

    # Obtain as many AM as class index
    output_cam = []
    for idx in class_idx:
        # Multiply the class FC weigths by the features
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))

        # Reshape, normalize and resize to create an image
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))

    return output_cam


def make_weights_for_balanced_classes(images, nclasses):
    """
    Function to obtain dataset dependent weights for the dataloader
    :param images:
    :param nclasses:
    :return:
    """
    count = [0] * nclasses
    for item in images:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))

    for i in range(nclasses):
        if count[i] is not 0:
            weight_per_class[i] = N/float(count[i])
        else:
            weight_per_class[i] = 1

    # weight_per_class_norm = [x / max(weight_per_class) for x in weight_per_class]

    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val]
    return weight


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

