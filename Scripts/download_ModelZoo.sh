#!/bin/bash

MODEL_DIR="./Data/Model Zoo"

# ADE20K Models
# Ours
wget -O "$MODEL_DIR/ADEChallengeData2016/SAScene_ResNet18_ADE.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/ADE20K/SAScene_ResNet18_ADE.pth.tar

# RGB Branch
wget -O "$MODEL_DIR/ADEChallengeData2016/RGB_ResNet18_ADE.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/ADE20K/RGB_ResNet18_ADE.pth.tar

# Semantic Branch
wget -O "$MODEL_DIR/ADEChallengeData2016/SemBranch_ADE.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/ADE20K/SemBranch_ADE.pth.tar

echo ========================================================================
echo "ADE20K Models saved to: "
echo -e "\033[32m $MODEL_DIR/ADEChallengeData2016/ \033[00m"
echo ========================================================================

# MIT Indoor 67 Models
# Ours. Backbone ResNet-18
wget -O "$MODEL_DIR/MITIndoor67/SAScene_ResNet18_MIT.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/MIT_Indoor_67/SAScene_ResNet18_MIT.pth.tar

# Ours. Backbone ResNet-50
wget -O "$MODEL_DIR/MITIndoor67/SAScene_ResNet50_MIT.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/MIT_Indoor_67/SAScene_ResNet50_MIT.pth.tar

# RGB Branch. Backbone ResNet-18
wget -O "$MODEL_DIR/MITIndoor67/RGB_ResNet18_MIT.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/MIT_Indoor_67/RGB_ResNet18_MIT.pth.tar

# RGB Branch. Backbone ResNet-50
wget -O "$MODEL_DIR/MITIndoor67/RGB_ResNet50_MIT.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/MIT_Indoor_67/RGB_ResNet50_MIT.pth.tar

# Semantic Branch
wget -O "$MODEL_DIR/MITIndoor67/SemBranch_MIT.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/MIT_Indoor_67/SemBranch_MIT.pth.tar

echo ========================================================================
echo "MIT Indoor 67 Models saved to: "
echo -e "\033[32m $MODEL_DIR/MITIndoor67/ \033[00m"
echo ========================================================================

# SUN 397 Models
# Ours. Backbone ResNet-18
wget -O "$MODEL_DIR/SUN397/SAScene_ResNet18_SUN.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/SUN397/SAScene_ResNet18_SUN.pth.tar

# Ours. Backbone ResNet-50
wget -O "$MODEL_DIR/SUN397/SAScene_ResNet50_SUN.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/SUN397/SAScene_ResNet50_SUN.pth.tar

# RGB Branch. Backbone ResNet-18
wget -O "$MODEL_DIR/SUN397/RGB_ResNet18_SUN.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/SUN397/RGB_ResNet18_SUN.pth.tar

# RGB Branch. Backbone ResNet-50
wget -O "$MODEL_DIR/SUN397/RGB_ResNet50_SUN.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/SUN397/RGB_ResNet50_SUN.pth.tar

# Semantic Branch
wget -O "$MODEL_DIR/SUN397/SemBranch_SUN.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/SUN397/SemBranch_SUN.pth.tar

echo ========================================================================
echo "SUN 397 Models saved to: "
echo -e "\033[32m $MODEL_DIR/SUN397/ \033[00m"
echo ========================================================================

# Places 365 Models
# Ours
wget -O "$MODEL_DIR/places365_standard/SAScene_ResNet18_Places.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/Places_365/SAScene_ResNet18_Places.pth.tar

# RGB Branch
wget -O "$MODEL_DIR/places365_standard/RGB_ResNet18_Places.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/Places_365/RGB_ResNet18_Places.pth.tar

# Semantic Branch
wget -O "$MODEL_DIR/places365_standard/SemBranch_Places.pth.tar" http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/Places_365/SemBranch_Places.pth.tar

echo ========================================================================
echo "Places 365 Models saved to: "
echo -e "\033[32m $MODEL_DIR/places365_standard/ \033[00m"
echo ========================================================================
