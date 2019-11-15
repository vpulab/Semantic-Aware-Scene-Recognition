#!/bin/bash

MODEL_DIR="./Data/Model Zoo"

# ADE20K Models
# Ours
wget -O "$MODEL_DIR/ADEChallengeData2016/SAScene_ResNet18_ADE.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EdjBkchzYmxMtemweugTNp8BAK_kzUmHwvFbmXgsE_VKRQ?download=1

# RGB Branch
wget -O "$MODEL_DIR/ADEChallengeData2016/RGB_ResNet18_ADE.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EY-boXwjIhZMq0LkaIqng48BY-ezHhF4t-0ctwOmSUYAjw?download=1

# Semantic Branch
wget -O "$MODEL_DIR/ADEChallengeData2016/SemBranch_ADE.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EYJ593CJEylMkRoyAO_UCWUBGKbUJIpGO_1VzeHeQYvBEA?download=1

echo ========================================================================
echo "ADE20K Models saved to: "
echo -e "\033[32m $MODEL_DIR/ADEChallengeData2016/ \033[00m"
echo ========================================================================

# MIT Indoor 67 Models
# Ours. Backbone ResNet-18
wget -O "$MODEL_DIR/MITIndoor67/SAScene_ResNet18_MIT.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EY5d6PU_Jo9ElazZpYvGn5cBI6aZChWQiyC3pXzey6L3cA?download=1

# Ours. Backbone ResNet-50
wget -O "$MODEL_DIR/MITIndoor67/SAScene_ResNet50_MIT.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EdM4OL-XCzxLpX8YEQ02msUBeEV1Swax0u5Gws6TKtcibw?download=1

# RGB Branch. Backbone ResNet-18
wget -O "$MODEL_DIR/MITIndoor67/RGB_ResNet18_MIT.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EQfMFvxYdIFPmY6jJ4OPCssB8axLM9KyW7JWGIoOVkF0oQ?download=1

# RGB Branch. Backbone ResNet-50
wget -O "$MODEL_DIR/MITIndoor67/RGB_ResNet50_MIT.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EUR0-OHoGOZIhh5ae5mCiFEBnsXJ3EJe93Kb4KfPvUMmGQ?download=1

# Semantic Branch
wget -O "$MODEL_DIR/MITIndoor67/SemBranch_MIT.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/ETRa2iuWq-BKqyEKTnSY_VkBzsO2FbSZvTyav5fi5iDpug?download=1

echo ========================================================================
echo "MIT Indoor 67 Models saved to: "
echo -e "\033[32m $MODEL_DIR/MITIndoor67/ \033[00m"
echo ========================================================================

# SUN 397 Models
# Ours. Backbone ResNet-18
wget -O "$MODEL_DIR/SUN397/SAScene_ResNet18_SUN.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EcNjUGAgE1dCss00_6A05_oBMjSUviEYigm0F_QcmW914g?download=1

# Ours. Backbone ResNet-50
wget -O "$MODEL_DIR/SUN397/SAScene_ResNet50_SUN.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EZxsF-jz-lJJlKyDg0aJZp0BlKHiEa3vszzc5UYwuRCVSg?download=1

# RGB Branch. Backbone ResNet-18
wget -O "$MODEL_DIR/SUN397/RGB_ResNet18_SUN.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/ERFBOWinVolPuYkAjdipF00BHxeQ9mjzlO5Oc_x3NLzDdw?download=1

# RGB Branch. Backbone ResNet-50
wget -O "$MODEL_DIR/SUN397/RGB_ResNet50_SUN.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EehUiiT53QtAr_NF74Rlk7gB1xaBhvppctChoALhMS5cCg?download=1

# Semantic Branch
wget -O "$MODEL_DIR/SUN397/SemBranch_SUN.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EQxpscYkhY5Nh55sdSgaINkBcqSMZ9b32K8AbrDfUKO2_w?download=1

echo ========================================================================
echo "SUN 397 Models saved to: "
echo -e "\033[32m $MODEL_DIR/SUN397/ \033[00m"
echo ========================================================================

# Places 365 Models
# Ours
wget -O "$MODEL_DIR/places365_standard/SAScene_ResNet18_Places.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/Eco7I0NbpWFFpJdYjT1om38BX2aEBR1WXXzflE2YykT0qA?download=1

# RGB Branch
wget -O "$MODEL_DIR/places365_standard/RGB_ResNet18_Places.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/EUaPpF5MpkNCqDd04GMVnYEB6BnH5inElN9ve_trAiWg-A?download=1

# Semantic Branch
wget -O "$MODEL_DIR/places365_standard/SemBranch_Places.pth.tar" https://dauam-my.sharepoint.com/:u:/g/personal/alejandro_lopezc01_estudiante_uam_es/Ea_B3l8vdRtJg3fjhwm7KeIBCbDGV5L2MOtpo9E5GGBo3Q?download=1

echo ========================================================================
echo "Places 365 Models saved to: "
echo -e "\033[32m $MODEL_DIR/places365_standard/ \033[00m"
echo ========================================================================