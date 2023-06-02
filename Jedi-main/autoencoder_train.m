clear;

%===================================================================
% Preparing training data:
% The training data (patch_gt) is a structure that contains binary 
% images of patch ground truth data where pixels that form the patch
% data have the value of 1 (white) and the other image data has 
% a value of 0 (black)
%===================================================================


% put the path to the patch ground truth training data here
patch_gt = load('');
patch_gt = patch_gt.patch_gt;

patch_gt = struct2cell(patch_gt);

hiddenSize1 = 100;

autoenc1 = trainAutoencoder(patch_gt,hiddenSize1, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false, ...
    'UseGPU',false);
	
%put the save path here
save_path = ''
save(save_path,autoenc1);
