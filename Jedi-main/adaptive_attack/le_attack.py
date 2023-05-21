import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import tqdm
from PIL import Image

import argparse
import csv
import os
import numpy as np
import random

from matplotlib.pyplot import imshow
from numpy.random import default_rng

from patch_utils import*
from utils import*

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
parser.add_argument('--train_size', type=int, default=4017, help="number of training images")
parser.add_argument('--test_size', type=int, default=100, help="number of test images")
parser.add_argument('--noise_percentage', type=float, default=0.05, help="percentage of the patch size compared with the image size")
parser.add_argument('--probability_eshold', type=float, default=0.9, help="minimum target probability")
parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
parser.add_argument('--max_iteration', type=int, default=1000, help="max iteration")
parser.add_argument('--target', type=int, default=5, help="target label")
parser.add_argument('--epochs', type=int, default=20, help="total epoch")
parser.add_argument('--data_dir', type=str, default='./data', help="dir of the dataset")
parser.add_argument('--patch_type', type=str, default='rectangle', help="type of the patch")
parser.add_argument('--GPU', type=str, default='0', help="index pf used GPU")
parser.add_argument('--log_dir', type=str, default='patch_attack_log.csv', help='dir of the log')
args = parser.parse_args()

# Patch attack via optimization
# According to reference [1], one image is attacked each time
# Assert: applied patch should be a numpy
# Return the final perturbated picture and the applied patch. Their types are both numpy
def patch_attack(image, applied_patch, mask, target, probability_eshold, model, lr=1, max_iteration=100):
    model.eval()
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    target_probability, count = 0, 0
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    while target_probability < probability_eshold and count < max_iteration:
        count += 1
        # Optimize the patch
        perturbated_image = Variable(perturbated_image.data, requires_grad=True)
        per_image = perturbated_image
        per_image = per_image.cuda()
        output = model(per_image)
        target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]
        target_log_softmax.backward()
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()
        applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor)
        applied_patch = torch.clamp(applied_patch, min=-3, max=3)
        # Test the patch
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
        perturbated_image = perturbated_image.cuda()
        output = model(perturbated_image)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]
    perturbated_image = perturbated_image.cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return perturbated_image, applied_patch

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

# Load the model
model = torch.load("./model/resnet50_pas07.pth")
model.cuda()
model.eval()


# Load the datasets
train_loader, test_loader = dataloader(args.train_size, args.test_size, args.data_dir, args.batch_size, args.num_workers, 4117)

# Test the accuracy of model on trainset and testset
trainset_acc, test_acc = test(model, train_loader), test(model, test_loader)
print('Accuracy of the model on clean trainset and testset is {:.3f}% and {:.3f}%'.format(100*trainset_acc, 100*test_acc))

# Initialize the patch
patch = patch_initialization(args.patch_type, image_size=(3, 224, 224), noise_percentage=args.noise_percentage)
print('The shape of the patch is', patch.shape)

with open(args.log_dir, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_success", "test_success"])

best_patch_epoch, best_patch_success_rate = 0, 0

#plot variables

rng = default_rng()
entropies = [0]
patch_performance = [0]
entr_limit = 6
entr_limit_plot = [entr_limit]
entr_max = [8]
xi = []
n_colors = 10

# Generate the patch
for epoch in range(args.epochs):
    train_total, train_actual_total, train_success = 0, 0, 0
    img_count = 0
    for (image, label) in tqdm.tqdm(train_loader, position=0, leave=True):
        train_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        #  predicted[0] != label
        if True and predicted[0].data.cpu().numpy() != args.target:
             train_actual_total += 1
             applied_patch, mask, x_location, y_location = mask_generation(args.patch_type, patch, image_size=(3, 224, 224))
             perturbated_image, applied_patch = patch_attack(image, applied_patch, mask, args.target, args.probability_eshold, model, args.lr, args.max_iteration)
             perturbated_image = torch.from_numpy(perturbated_image).cuda()
             output = model(perturbated_image)
             _, predicted = torch.max(output.data, 1)
             if predicted[0].data.cpu().numpy() == args.target:
                 train_success += 1
             patch = applied_patch[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
    
        img_count = img_count + 1
        
        freq = 400
        label_it = 0
        
        if img_count % (freq // 4) == 0:
            test_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
            
            #prepare plot parameters
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            patch_np = np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1)
            patch_pil = Image.fromarray(np.uint8(patch_np*255))
            entropy = patch_pil.convert('L').entropy()
            
            entropies.append(entropy)
            patch_performance.append(test_success_rate * 100)
            entr_limit_plot.append(6.01)
            entr_max.append(8)
            
            
            
            plt.xlabel('iteration #')
            plt.ylabel('entropy')
            if label_it % 4 == 0:
                xi.append(freq * label_it)
            else:
                xi.append('')
            plt.plot(entropies,'r',label = 'Patch entropy')
            plt.plot(entr_limit_plot,'b', label = 'Natural image entropy')
            plt.plot(entr_max,'y', label = 'Random noise entropy')
            plt.xticks(list(range(len(entr_limit_plot))),xi[0:len(entr_limit_plot)])
            plt.legend()
            plt.show()
        
        
        if img_count % freq == 0:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success / train_actual_total))
            test_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
            print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, 100 * test_success_rate))
        
            # Record the statistics
            with open(args.log_dir, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, test_success_rate])
        
            if test_success_rate > best_patch_success_rate:
                best_patch_success_rate = test_success_rate
                best_patch_epoch = epoch


            #entropy calc
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            patch_np = np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1)
            patch_pil = Image.fromarray(np.uint8(patch_np*255))
            entropy = patch_pil.convert('L').entropy()
            
            e_i = entropy
            t_i = test_success_rate
            
            #VNS
            if entropy > entr_limit:
                
                #variable initialization for color vns
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                patch_np = np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1)
                patch_pil = Image.fromarray(np.uint8(patch_np*255))
                patch_array = np.array(patch_pil)[:,:,0:3]
                patch_gr = transforms.Grayscale()(patch_pil)
                patch_array_gr = np.array(patch_gr)
                vns_entropy = entropy
                it = 0
                
                while vns_entropy > entr_limit:
                    #current iteration variables
                    it = it +1
                    vns_patches = []
                    vns_entropies = []
                    vns_ASR = []
                    values_to_remove = random.sample(range(256),n_colors)
                    
                    for v in range(n_colors):
                        #find the positions of the colors to replace
                        vns_temp_patch_array = patch_array
                        vns_temp_patch = patch
                        pixel_list = np.array([[0,0]])
                        pixel_list = np.argwhere(patch_array_gr == values_to_remove[v])
                        
                        if len(pixel_list) == 0:
                            print("No pixels vith value {}".format(values_to_remove[v]))
                            vns_patches.append(vns_temp_patch)
                            vns_entropies.append(999)
                            vns_ASR.append(0)
                        else:
                            for px in range(len(pixel_list)):
                                
                                #find the closest color
                                dist = vns_temp_patch_array - vns_temp_patch_array[pixel_list[px,0],pixel_list[px,1]]
                                sqr0 = np.square(dist[:,:,0],dtype="uint32")
                                sqr1 = np.square(dist[:,:,1],dtype="uint32")
                                sqr2 = np.square(dist[:,:,2],dtype="uint32")
                                dist_e = np.sqrt(sqr1 + sqr2 + sqr0)
                                
                                min_dist = np.min(dist_e[np.nonzero(dist_e)])
                                closest_color_pos = np.argwhere(dist_e == min_dist)
                                
                                #replace the pixel
                                vns_temp_patch[:,pixel_list[px,0],pixel_list[px,1]] = vns_temp_patch[:,closest_color_pos[0,0],closest_color_pos[0,1]]
                                
                            #calculate entropy and ASR of the new patch
                            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                            vns_temp_patch_np = np.clip(np.transpose(vns_temp_patch, (1, 2, 0)) * std + mean, 0, 1)
                            vns_temp_patch_pil = Image.fromarray(np.uint8(vns_temp_patch_np*255))
                            vns_temp_entr = vns_temp_patch_pil.convert('L').entropy()
                            vns_temp_ASR = test_patch(args.patch_type, args.target, vns_temp_patch, test_loader, model)
                            
                            vns_patches.append(vns_temp_patch)
                            vns_entropies.append(vns_temp_entr)
                            vns_ASR.append(vns_temp_ASR)
                        
                    #find the best patch among the n random removed colors
                    best_vns_pos = np.argmax(vns_ASR)
                    #best_vns_pos = np.argmin(vns_entropies)
                    
                    #display VNS progress data
                    print()
                    print("Epoch {}, iteration {}, VNS step {}".format(epoch,img_count,it))
                    print("Found a new best patch: {}".format(best_vns_pos))
                    print("New entropy: {:.3f}, was {:.3f}".format(vns_entropies[best_vns_pos],vns_entropy))
                    print("New ASR: {:.3f}%, was {:.3f}%".format(vns_ASR[best_vns_pos] * 100,test_success_rate * 100))
                    
                    #show patch
                    
                    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    patch_npt = np.clip(np.transpose(vns_patches[best_vns_pos], (1, 2, 0)) * std + mean, 0, 1)
                    patch_pilt = Image.fromarray(np.uint8(patch_npt*255))
                    #plt.imshow(patch_pilt)
                    
                    #Replace the old patch
                    
                    patch = vns_patches[best_vns_pos]
                    vns_entropy = vns_entropies[best_vns_pos]
                    entropy = vns_entropy
                    test_success_rate = vns_ASR[best_vns_pos]
                    
                    
                
                                
            #plot entropy/performance
            entropies.append(entropy)
            #patch_performance.append(test_success_rate * 100)
            entr_limit_plot.append(6.01)
            entr_max.append(8)
            plt.xlabel('iteration #')
            plt.ylabel('entropy')
            xi = np.multiply(list(range(len(entr_limit_plot))),100)
            #plt.plot(patch_performance,'b')
            plt.plot(entropies,'r')
            plt.plot(entr_limit_plot,'y')
            plt.xticks(list(range(len(entr_limit_plot))),xi)
            plt.show()
            
            #save the patch
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            patch_npt = np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1)
            patch_pilt = Image.fromarray(np.uint8(patch_npt*255))
            ptch_path = "./le_patch/" + "epoch_" + str(epoch) + "_iter_" + str(img_count) + ".png"
            patch_pilt.save(ptch_path)
 
                    
                

    # Load the statistics and generate the line
    #log_generation(args.log_dir)
    
    train_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
    print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success_rate))

print("The best patch is found at epoch {} with success rate {}% on testset".format(best_patch_epoch, 100 * best_patch_success_rate))
