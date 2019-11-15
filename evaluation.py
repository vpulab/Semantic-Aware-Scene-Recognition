"""
Evalaution file to completly test the trained model
Usage:
    --ConfigPath [PATH to configuration file for desired dataset]
"""


import os
import time
import torch
import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from RGBBranch import RGBBranch
from SemBranch import SemBranch
from SASceneNet import SASceneNet
from Libs.Datasets.ADE20KDataset import ADE20KDataset
from Libs.Datasets.MITIndoor67Dataset import MITIndoor67Dataset
from Libs.Datasets.SUN397Dataset import SUN397Dataset
from Libs.Datasets.Places365Dataset import Places365Dataset
from Libs.Utils import utils
from Libs.Utils.torchsummary import torchsummary
import numpy as np
import yaml


parser = argparse.ArgumentParser(description='Semantic-Aware Scene Recognition Evaluation')
parser.add_argument('--ConfigPath', metavar='DIR', help='Configuration file path')


def evaluationDataLoader(dataloader, model, set):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top2 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    ClassTPs_Top1 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    ClassTPs_Top2 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    ClassTPs_Top5 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    Predictions = np.zeros(len(dataloader))
    SceneGTLabels = np.zeros(len(dataloader))

    # Extract batch size
    batch_size = CONFIG['VALIDATION']['BATCH_SIZE']['TEST']

    # Start data time
    data_time_start = time.time()

    with torch.no_grad():
        for i, (mini_batch) in enumerate(dataloader):
            start_time = time.time()
            if USE_CUDA:
                RGB_image = mini_batch['Image'].cuda()
                semantic_mask = mini_batch['Semantic'].cuda()
                semantic_scores = mini_batch['Semantic Scores'].cuda()
                sceneLabelGT = mini_batch['Scene Index'].cuda()

            if set is 'Validation' and CONFIG['VALIDATION']['TEN_CROPS']:
                # Fuse batch size and ncrops to set the input for the network
                bs, ncrops, c_img, h, w = RGB_image.size()
                RGB_image = RGB_image.view(-1, c_img, h, w)

                bs, ncrops, c_sem, h, w = semantic_mask.size()
                semantic_mask = semantic_mask.view(-1, c_sem, h, w)

                bs, ncrops, c_sem, h, w = semantic_scores.size()
                semantic_scores = semantic_scores.view(-1, c_sem, h, w)

            # Create tensor of probabilities from semantic_mask
            semanticTensor = utils.make_one_hot(semantic_mask, semantic_scores, C=CONFIG['DATASET']['N_CLASSES_SEM'])

            # Model Forward
            outputSceneLabel, feature_conv, outputSceneLabelRGB, outputSceneLabelSEM = model(RGB_image, semanticTensor)

            if set is 'Validation' and CONFIG['VALIDATION']['TEN_CROPS']:
                # Average results over the 10 crops
                outputSceneLabel = outputSceneLabel.view(bs, ncrops, -1).mean(1)
                outputSceneLabelRGB = outputSceneLabelRGB.view(bs, ncrops, -1).mean(1)
                outputSceneLabelSEM = outputSceneLabelSEM.view(bs, ncrops, -1).mean(1)

            if batch_size is 1:
                if set is 'Validation' and CONFIG['VALIDATION']['TEN_CROPS']:
                    feature_conv = torch.unsqueeze(feature_conv[4, :, :, :], 0)
                    RGB_image = torch.unsqueeze(RGB_image[4, :, :, :], 0)

                # Obtain 10 most scored predicted scene index
                Ten_Predictions = utils.obtainPredictedClasses(outputSceneLabel)

                # Save predicted label and ground-truth label
                Predictions[i] = Ten_Predictions[0]
                SceneGTLabels[i] = sceneLabelGT.item()

                # Compute activation maps
                # utils.saveActivationMap(model, feature_conv, Ten_Predictions, sceneLabelGT,
                #                         RGB_image, classes, i, set, save=True)

            # Compute class accuracy
            ClassTPs = utils.getclassAccuracy(outputSceneLabel, sceneLabelGT, len(classes), topk=(1, 2, 5))
            ClassTPs_Top1 += ClassTPs[0]
            ClassTPs_Top2 += ClassTPs[1]
            ClassTPs_Top5 += ClassTPs[2]

            # Compute Loss
            loss = model.loss(outputSceneLabel, sceneLabelGT)

            # Measure Top1, Top2 and Top5 accuracy
            prec1, prec2, prec5 = utils.accuracy(outputSceneLabel.data, sceneLabelGT, topk=(1, 2, 5))

            # Update values
            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)
            top2.update(prec2.item(), batch_size)
            top5.update(prec5.item(), batch_size)

            # Measure batch elapsed time
            batch_time.update(time.time() - start_time)

            # Print information
            if i % CONFIG['VALIDATION']['PRINT_FREQ'] == 0:
                print('Testing {} set batch: [{}/{}]\t'
                      'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} (avg: {loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f})\t'
                      'Prec@2 {top2.val:.3f} (avg: {top2.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} (avg: {top5.avg:.3f})'.
                      format(set, i, len(dataloader), set, batch_time=batch_time, loss=losses,
                             top1=top1, top2=top2, top5=top5))

        ClassTPDic = {'Top1': ClassTPs_Top1.cpu().numpy(),
                      'Top2': ClassTPs_Top2.cpu().numpy(), 'Top5': ClassTPs_Top5.cpu().numpy()}

        print('Elapsed time for {} set evaluation {time:.3f} seconds'.format(set, time=time.time() - data_time_start))
        print("")

        if batch_size is 1:
            # Save predictions and Scene GT in txt file
            np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/' + set + '_Predictions.txt', np.transpose(Predictions), '%i')
            np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/' + set + '_GT.txt', np.transpose(SceneGTLabels), '%i')

        return top1.avg, top2.avg, top5.avg, losses.avg, ClassTPDic


global USE_CUDA, classes, CONFIG

# Decode CONFIG file information
args = parser.parse_args()
CONFIG = yaml.safe_load(open(args.ConfigPath, 'r'))
USE_CUDA = torch.cuda.is_available()

print('-' * 65)
print("Evaluation starting...")
print('-' * 65)


# Instantiate network
if CONFIG['MODEL']['ONLY_RGB']:
    print('Evaluating ONLY RGB branch')
    print('Selected RGB backbone architecture: ' + CONFIG['MODEL']['ARCH'])
    model = RGBBranch(arch=CONFIG['MODEL']['ARCH'], scene_classes=CONFIG['DATASET']['N_CLASSES_SCENE'])
elif CONFIG['MODEL']['ONLY_SEM']:
    print('Evaluating ONLY SEM branch')
    model = SemBranch(scene_classes=CONFIG['DATASET']['N_CLASSES_SCENE'], semantic_classes=CONFIG['DATASET']['N_CLASSES_SEM'])
else:
    print('Evaluating complete model')
    print('Selected RG backbone architecture: ' + CONFIG['MODEL']['ARCH'])
    model = SASceneNet(arch=CONFIG['MODEL']['ARCH'], scene_classes=CONFIG['DATASET']['N_CLASSES_SCENE'], semantic_classes=CONFIG['DATASET']['N_CLASSES_SEM'])


# Load the trained model
completePath = CONFIG['MODEL']['PATH'] + CONFIG['MODEL']['NAME'] + '.pth.tar'
if os.path.isfile(completePath):
    print("Loading model {} from path {}...".format(CONFIG['MODEL']['NAME'], completePath))
    checkpoint = torch.load(completePath)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded model {} from path {}.".format(CONFIG['MODEL']['NAME'], completePath))
    print("     Epochs {}".format(checkpoint['epoch']))
    print("     Single crop reported precision {}".format(best_prec1))
else:
    print("No checkpoint found at '{}'. Check configuration file MODEL field".format(completePath))
    quit()

# Move Model to GPU an set it to evaluation mode
if USE_CUDA:
    model.cuda()
cudnn.benchmark = USE_CUDA
model.eval()

print('-' * 65)
print('Loading dataset {}...'.format(CONFIG['DATASET']['NAME']))

traindir = os.path.join(CONFIG['DATASET']['ROOT'], CONFIG['DATASET']['NAME'])
valdir = os.path.join(CONFIG['DATASET']['ROOT'], CONFIG['DATASET']['NAME'])

if CONFIG['DATASET']['NAME'] == "ADEChallengeData2016":
    # Training Dataset
    train_dataset = ADE20KDataset(traindir, "training")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'],
                                               shuffle=False, num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], pin_memory=True)

    # Validation Dataset
    val_dataset = ADE20KDataset(valdir, "validation", tencrops=CONFIG['VALIDATION']['TEN_CROPS'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TEST'],
                                             shuffle=False, num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], pin_memory=True)

    classes = train_dataset.classes

elif CONFIG['DATASET']['NAME'] == "places365_standard":
    train_dataset = Places365Dataset(traindir, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'],
                                               shuffle=False, num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], pin_memory=True)

    val_dataset = Places365Dataset(valdir, "val", tencrops=CONFIG['VALIDATION']['TEN_CROPS'], SemRGB=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TEST'],
                                             shuffle=False, num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], pin_memory=True)

    classes = train_dataset.classes

elif CONFIG['DATASET']['NAME'] == "MITIndoor67":
    train_dataset = MITIndoor67Dataset(traindir, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'],
                                               shuffle=False, num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], pin_memory=True)

    val_dataset = MITIndoor67Dataset(valdir, "val", tencrops=CONFIG['VALIDATION']['TEN_CROPS'], SemRGB=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TEST'], shuffle=False,
                                             num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], pin_memory=True)

    classes = train_dataset.classes

elif CONFIG['DATASET']['NAME'] == "SUN397":
    train_dataset = SUN397Dataset(traindir, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'],
                                               shuffle=False, num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], pin_memory=True)

    val_dataset = SUN397Dataset(valdir, "val", tencrops=CONFIG['VALIDATION']['TEN_CROPS'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TEST'],
                                             shuffle=False, num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], pin_memory=True)

    classes = train_dataset.classes

# Print dataset information
print('Dataset loaded!')
print('Dataset Information:')
print('Train set. Size {}. Batch size {}. Nbatches {}'
      .format(len(train_loader) * CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'], CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'], len(train_loader)))
print('Validation set. Size {}. Batch size {}. Nbatches {}'
      .format(len(val_loader) * CONFIG['VALIDATION']['BATCH_SIZE']['TEST'], CONFIG['VALIDATION']['BATCH_SIZE']['TEST'], len(val_loader)))
print('Train set number of scenes: {}' .format(len(classes)))
print('Validation set number of scenes: {}' .format(len(classes)))

print('-' * 65)

print('Computing histogram of scene classes...')

TrainHist = utils.getHistogramOfClasses(train_loader, classes, "Training")
ValHist = utils.getHistogramOfClasses(val_loader, classes, "Validation")

# Check if OUTPUT_DIR exists and if not create it
if not os.path.exists(CONFIG['EXP']['OUTPUT_DIR']):
    os.makedirs(CONFIG['EXP']['OUTPUT_DIR'])

# Save Dataset histograms
np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/TrainingHist.txt', TrainHist, '%u')
np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/ValidationHist.txt', ValHist, '%u')

# Print Network information
print('-' * 65)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Number of params: {}'. format(params))
print('-' * 65)
print('GPU in use: {} with {} memory'.format(torch.cuda.get_device_name(0), torch.cuda.max_memory_allocated(0)))
print('-' * 65)

# Summary of the network for a dummy input
sample = next(iter(val_loader))
torchsummary.summary(model, [(3, 224, 224), (CONFIG['DATASET']['N_CLASSES_SEM'] + 1, 224, 224)], batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'])


print('Evaluating dataset ...')

# Evaluate model on training set
# train_top1, train_top2, train_top5, train_loss, train_ClassTPDic\
#     = evaluationDataLoader(train_loader, model, set='Training')
#
# # Save Training Class Accuracy
# train_ClassAcc_top1 = (train_ClassTPDic['Top1'] / (TrainHist + 0.0001)) * 100
# np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/TrainingTop1ClassAccuracy.txt', np.transpose(train_ClassAcc_top1), '%f')
#
# train_ClassAcc_top2 = (train_ClassTPDic['Top2'] / (TrainHist + 0.0001)) * 100
# np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/TrainingTop2ClassAccuracy.txt', np.transpose(train_ClassAcc_top2), '%f')
#
# train_ClassAcc_top5 = (train_ClassTPDic['Top5'] / (TrainHist + 0.0001)) * 100
# np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/TrainingTop5ClassAccuracy.txt', np.transpose(train_ClassAcc_top5), '%f')

# Evaluate model on validation set
val_top1, val_top2, val_top5, val_loss, val_ClassTPDic = evaluationDataLoader(val_loader, model, set='Validation')

# Save Validation Class Accuracy
val_ClassAcc_top1 = (val_ClassTPDic['Top1'] / (ValHist + 0.0001)) * 100
np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/ValidationTop1ClassAccuracy.txt', np.transpose(val_ClassAcc_top1), '%f')

val_ClassAcc_top2 = (val_ClassTPDic['Top2'] / (ValHist + 0.0001)) * 100
np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/ValidationTop2ClassAccuracy.txt', np.transpose(val_ClassAcc_top2), '%f')

val_ClassAcc_top5 = (val_ClassTPDic['Top5'] / (ValHist + 0.0001)) * 100
np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/ValidationTop5ClassAccuracy.txt', np.transpose(val_ClassAcc_top5), '%f')

# Print complete evaluation information
print('-' * 65)
print('Evaluation statistics:')

# print('Train results     : Loss {train_loss:.3f}, Prec@1 {top1:.3f}, Prec@2 {top2:.3f}, Prec@5 {top5:.3f}, '
#       'Mean Class Accuracy {MCA:.3f}'.format(train_loss=train_loss, top1=train_top1, top2=train_top2, top5=train_top5,
#                                               MCA=np.mean(train_ClassAcc_top1)))

print('Validation results: Loss {val_loss:.3f}, Prec@1 {top1:.3f}, Prec@2 {top2:.3f}, Prec@5 {top5:.3f}, '
      'Mean Class Accuracy {MCA:.3f}'.format(val_loss=val_loss, top1=val_top1, top2=val_top2, top5=val_top5,
                                              MCA=np.mean(val_ClassAcc_top1)))
