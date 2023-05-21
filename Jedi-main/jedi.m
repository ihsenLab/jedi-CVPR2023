% single folder inpainting code

clear;

%do you  want to use the autoencoder?
use_autoencoder = 1;
%do you wish to evaluate mask accuracy
do_eval = 1;

%define the paths for the adversarial images and the folder where you
%want the results here
adv_images_path = "./adversarial_samples/";
cleaned_images_path = "./cleaned_samples/";
adv_images_dir = dir(adv_images_path);

%define the ground truth path if you want to evaluate mask accuracy
if do_eval == 1
    patch_gt = load('./sample_gt.mat');
    patch_gt = patch_gt.patch_gt;
end

%define the autoencoder path if you wish to use it
if use_autoencoder == 1
    autoenc = load('./autoenc_pas07.mat');
    autoenc = autoenc.autoenc1;
end

masks = struct();

detect_ratio = zeros([1,size(adv_images_dir,1)-2]); 
psize = zeros([1,size(adv_images_dir,1)-2]); 

%entropy statistics from patches and known-clean images
p_mean = [5.32,6.06,6.47];
p_std = [0.32,0.34,0.35];

c_mean = [2.75,3.19,3.47];
c_std = [1.16,1.13,1.07];

entropy_values = struct('patch_mean',p_mean,'patch_stdev',p_std,...
                        'clean_mean',c_mean,'clean_stdev',c_std);

for im = 3 :size(adv_images_dir,1)
    
    fprintf("Image %d / %d \n ",im-2,size(adv_images_dir,1)-2)
    image_path = strcat(adv_images_path,adv_images_dir(im).name);
    img = imread(image_path);

    
    if size(img,3) == 1
        img = cat(3,img,img,img);
    end
    size_x = size(img,1);
    size_y = size(img,2);
    
    sx = ceil(size_x/100) + mod(ceil(size_x/100),2);
    sy = ceil(size_y/100) + mod(ceil(size_y/100),2);
    
    s1 = max(sx,sy);
    s2 = max(s1,8);
    
    ws = [s2,s2*1.5+mod(s2*1.5,2),s2*2];
    strd = ws/2;
    
    area = size_x * size_y;
    [ents,e_htmp_100] = get_entr_heatmap(img,size_x,size_y,ws(1),strd(1),entropy_values,1);
    [ents,e_htmp_150] = get_entr_heatmap(img,size_x,size_y,ws(2),strd(2),entropy_values,2);
    [ents,e_htmp_200] = get_entr_heatmap(img,size_x,size_y,ws(3),strd(3),entropy_values,3);
    
    e_htmp = e_htmp_100 + e_htmp_150 + e_htmp_200;
    
    
    %cleanup
    mask = e_htmp > 0;
    mask = mask(1:size_x,1:size_y);
    stats = regionprops(mask,'Area','BoundingBox','PixelIdxList');
    
    
    for blob = 1:size(stats,1)
        if (stats(blob).Area / area < 0.005 && (size(stats,1)>1 || max([stats.Area]) > 0.01))
            mask(stats(blob).PixelIdxList) = 0;
        end
    end
    
    if use_autoencoder == 1
        autoenc_mask =  predict(autoenc,mask);
        mask = autoenc_mask > 0.33; 
    end
    
    %debug: see marked areas
    %imshow(imfuse(img,mask))
    %pause(2)
    
    % calculate % of detected patch area
    if do_eval == 1
        fld_name = "gt" + strrep(adv_images_dir(im).name(1:end-4),"-","_");
        gt = patch_gt.(fld_name);
        patch_px = nnz(gt);

        overlap = gt & mask;
        overlap_px = nnz(overlap);

        intersection = gt & mask;
        intersection_px = nnz(intersection);

        union = gt | mask;
        union_px = nnz(union);

        iou = intersection_px / union_px;

        detect_ratio(im-2) = iou;
        psize(im-2) = sqrt(patch_px);
    end
    
    %inpainting + save
    new_img = mitigate_patch(img,mask);
    save_path = strcat(cleaned_images_path,adv_images_dir(im).name);
    imwrite(new_img,save_path);
    
end

%% heatmap function
function [ents,e_htmp] = get_entr_heatmap(img,size_x,size_y,winsize,strd,entropy_values,i)

img_gr = rgb2gray(img);
entr_heatmap = zeros(size(img_gr));

win_size_x = winsize;
win_size_y = winsize;

img_gr = padarray(img_gr,[win_size_x/2,win_size_y/2],'symmetric');

stride_x = strd;
stride_y = strd;

ents = [];
for x = 1 : stride_x : size(img_gr,1) - win_size_x
    for y = 1 : stride_y : size(img_gr,2) - win_size_y
        window_cur = img_gr(x:x+win_size_x,y:y+win_size_y);
        win_entr = entropy(window_cur);
        entr_heatmap(x:x+win_size_x,y:y+win_size_y) = win_entr;
        ents = [ents,win_entr];
    end
end
x_exc = size(entr_heatmap,1) - size_x;
y_exc = size(entr_heatmap,2) - size_y;
entr_heatmap = entr_heatmap(round(x_exc / 2):size_x+round(x_exc / 2)-1,round(y_exc / 2):size_y+round(y_exc / 2)-1);
emax =max(max(entr_heatmap));
emin=min(min(entr_heatmap));
ediff = emax - emin;


patch_mean = entropy_values.patch_mean(i);
patch_std = entropy_values.patch_stdev(i);

clean_mean = entropy_values.clean_mean(i);
clean_stdev = entropy_values.clean_stdev(i);

img_mean = mean(ents);
img_spread = (img_mean - clean_mean) / clean_stdev;

dev_mult = ((patch_mean - clean_mean) / patch_mean) * 2;

thr_base = (patch_mean + 1.5 * patch_std) - (dev_mult * patch_std);
entr_thr = thr_base + (img_spread * patch_std);

e_htmp = entr_heatmap - entr_thr;

%debug: show selected entropy threshold
%fprintf('ws = %d , thr = %f \n',win_size_x,entr_thr)

e_htmp = max(e_htmp,0);

e_htmp = e_htmp ~= 0;

end

%% patch mitigation
function new_img = mitigate_patch(img,mask)
[r,c] = ind2sub(size(mask),find(mask == 1));
for ind = 1 : size(r)
    img(r(ind),c(ind),:) = [128,128,128];
end
new_img = inpaintCoherent(img,mask);
end