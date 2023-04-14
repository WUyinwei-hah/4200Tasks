import os
import cv2
import numpy as np
# this file is use to generate patch dataset

########## TODO: set parameters ##################
# set your 6k 4k images path
dataset_origin_resolution_path = "/home/Wuyinwei/Desktop/DehazingWeatherDiffusion/NTIRE"

# set your patch images save path
dataset_root_path = "test_patch"

# set your images and patches parameters
w = 6000
h = 4000
# your patch size.
patch_w = 600
patch_h = 400
# How many pathes
num_w = 7
num_h = 7
#################################################


x = np.linspace(0, w-patch_w, num_w)
y = np.linspace(0, h-patch_h, num_h)
X, Y = np.meshgrid(x, y)
print(x)
print(y)



# get HR images path
hr_train_path = os.path.join(dataset_origin_resolution_path, 'train')
hr_val_path = os.path.join(dataset_origin_resolution_path, 'val')
hr_test_path = os.path.join(dataset_origin_resolution_path, 'test')


for type in ["test", "train", "val"]:
    input_names = []
    path_save = os.path.join(dataset_origin_resolution_path, type)

    for root, dirs, files in os.walk(os.path.join(path_save, 'hazy'), topdown=False):
        for name in files:
            input_names.append(os.path.join(root, name))


    patch_data_save_patch = os.path.join(dataset_root_path, type)

    for path in input_names:

        origin_img  = cv2.imread(path)
        h, w, _ = origin_img.shape
        gt_img_path = path.replace('/hazy/', '/GT/')
        if type == "test":
            gt_img = None
        else:
            gt_img  = cv2.imread(gt_img_path)

        if h > w:
            origin_img =  cv2.rotate(origin_img, cv2.ROTATE_90_CLOCKWISE)
            # if test set, gt_img is None 
            if gt_img is not None:
                gt_img =  cv2.rotate(gt_img, cv2.ROTATE_90_CLOCKWISE)

        img_name = path.split("/")[-1]

        hazy_path = os.path.join(patch_data_save_patch, "hazy")
        GT_path = os.path.join(patch_data_save_patch, "GT")
        
        if not os.path.exists(hazy_path):
            os.makedirs(hazy_path)
        if not os.path.exists(GT_path):
            os.makedirs(GT_path)

        for i in range(len(x)):
            for j in range(len(y)):
                h_i = int(y[j])
                w_i = int(x[i])
                patch_input = origin_img[h_i:h_i+patch_h, w_i:w_i+patch_w, :]

                if gt_img is not None:
                    patch_gt = gt_img[h_i:h_i+patch_h, w_i:w_i+patch_w, :]

                input_save_patch = os.path.join(patch_data_save_patch, "hazy", f"{img_name}_{i}_{j}.jpg")
                gt_save_patch = os.path.join(patch_data_save_patch, "GT", f"{img_name}_{i}_{j}.jpg")

                cv2.imwrite(input_save_patch, patch_input)
                if gt_img is not None:
                    cv2.imwrite(gt_save_patch, patch_gt)
                
    

