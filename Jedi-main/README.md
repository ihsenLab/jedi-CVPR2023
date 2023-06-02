# Jedi Adversarial Patch Defense
This is the anonymous repository for: "Jedi: Entropy-based Localization and Removal of Adversarial Patches"

## Running Jedi

Jedi.m is the main file used for detecting and removing adversarial patches. To run this file, follow the next steps

Step 1 : Set whether you wish to use the autoencoder or evaluate the accuracy of the generated masks in lines 5 and 7 respectively

Step 2 : Indicate the folders where the adversarial images are (line 12) and where the cleaned images will be saved (line 14)

Step 3 : If you wish to evaluate the mask accuracy, change the path in line 18 to the mat file containing the ground truths. Otherwise ignore this step.

Step 4: If you wish to use the autoencoder, change the path in line 24 to file containing the autoencoder, Otherwise ignore this step

### Autoencoder

We provide pre-trained auto encoders that we used for our experiments:

* Imagenet and Pascal VOC 07: [Auto encoder link](https://drive.google.com/file/d/1N3BXaWu85uNJ378_SHkU_HElAIiaCaDr/view?usp=sharing) . This is the auto encoder to use with the provided sample images. This auto encoder should also function with any 224x224 pixel image
* CASIA : [Auto encoder link](https://drive.google.com/file/d/1AO4fYILPCQlGO15PrbGV8JS2RgoHXz-X/view?usp=sharing) The image size for this autoencoder is 320x240

**Training your own auto encoder:**

If you wish to train an auto encoder for your own custom data, it is possible to do so via the "autoencoder_train.m" file. The training data (patch_gt) is a matlab structure that contains binary images of patch ground truth data where pixels that form the patch data have the value of 1 (white) and the other image data has a value of 0 (black)

## Evaluation
The evaluuation files require Python 3.7 and Pytorch

For the classification evaluations, weight files for the Pascal VOC 07 dataset are available here

[Inception_v3](https://drive.google.com/file/d/19uROGUGR71wdu-kbLi63p-fFm97U5e5i/view?usp=sharing)

[ResNet-18](https://drive.google.com/file/d/1EfTd0pFohg_61UGKqxZCBC9Pe8xF4KZy/view?usp=sharing)

[ResNet-50](https://drive.google.com/file/d/1sDMWhx90ft2iwUW8Be7YbBGHHiyCNYbn/view?usp=sharing)

[ResNet-101](https://drive.google.com/file/d/1tzRJzmYpOH5LqN0VOqfxsgRLuNluSmlS/view?usp=sharing)

[VGG-16](https://drive.google.com/file/d/1yzow_A_5GEugfWjjF6KsH1hgPwy28x9Z/view?usp=sharing)


[VGG-19](https://drive.google.com/file/d/1bUeuWAyIQotmashylJCPXL9noXvt-VJm/view?usp=sharing)

For the detection evaluation, the darknet repository is required [(here)](https://github.com/pjreddie/darknet)

Yolov2/3/4 weights are available here

[YoloV2](https://drive.google.com/file/d/1iEm6tv521flagCJzwUy2KURTFH1O7zcA/view?usp=sharing)

[YoloV3](https://drive.google.com/file/d/120vF6NEcUSpTNXovsRA6oeIk3LR_zxk0/view?usp=sharing)

[YoloV4](https://drive.google.com/file/d/1V_xNETpN4Tq6w-wBnBPGdkYhNhR_xpSy/view?usp=sharing)

## Adaptive attack

Our adaptive attack is based on [this](https://github.com/A-LinCui/Adversarial_Patch_Attack/) implementation of [Adversarial Patch](https://arxiv.org/abs/1712.09665) by Brown et al. 

Add le_attack.py to the root of the linked repository and run it after checking the setting and variables in order to generate a low entropy patch.
