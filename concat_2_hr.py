from ntire import NTIREDownDataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch
import os
import torchvision
from torchvision.transforms.functional import crop
from collections import OrderedDict
from models import *

def overlapping_grid_indices(x_cond, output_size, n_w=30, n_h=20):
        _, c, h, w = x_cond.shape
        w_list = np.linspace(0, w-output_size, n_w, dtype=int)
        h_list = np.linspace(0, h-output_size, n_h, dtype=int)
        return h_list, w_list

def get_result(img_batch, y, model, p_size=256, device=torch.device("cuda:6"), n_w=2, n_h=1):
    # img_batch: 输入的一个batch的图片, (b, c, h, w) 的 shape，一般dataloader加载进来就这样
    # y: 标签
    # p_size: 一个patch的大小，一般默认为正方形
    # n_w: 沿着宽度方向切多少份
    # n_h: 沿着长度方向切多少份


    _, c, h, w = img_batch.shape
    rotate = False

    # 横着放
    if h > w:
        img_batch =  torch.rot90(img_batch, 1, [2, 3])
        rotate = True
    img_batch = img_batch.to(device)

    # 检查格式
    if img_batch.shape[1] == 4:
        img_batch = img_batch[:, :3, :, :]
    
    output = torch.zeros_like(img_batch, device=device)
    x_grid_mask = torch.zeros_like(img_batch, device=device)
    h_list, w_list = overlapping_grid_indices(img_batch, output_size=256, n_w=n_w, n_h=n_h)
    corners = [(i, j) for i in h_list for j in w_list]

    for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

    # 手动设置patch size
    manual_batching_size = 75

    with torch.no_grad():

        # 把每个输入切割成指定patch的大小
        temp_batch = torch.cat([crop(img_batch, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)

        for i in range(0, len(corners), manual_batching_size):
            print(f"start {i} to {i+manual_batching_size}")
            outputs = model(temp_batch[i:i+manual_batching_size].to(device))
            for idx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
                output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
        
        # 取平均
        # # 1, 3, 4000, 6000
        final_output = torch.div(output, x_grid_mask).clamp_(-1, 1)
        if rotate:
            final_output = torch.rot90(final_output, 3, [2, 3]).squeeze()
        final_output = final_output * 0.5 + 0.5
        torchvision.utils.save_image(final_output, os.path.join("./experiment_output/", y+f'_{n_w}_{n_h}_.jpg'))

def single(save_dir, device):
	state_dict = torch.load(save_dir, device)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict

def main():
    test_dataset = NTIREDataset(mode="test")

    test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=4,
							 pin_memory=True)
    device = torch.device("cuda:6") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    model_name = "gunet_d"
    network = eval(model_name.replace('-', '_'))()
    network.cuda(device)
    saved_model_dir = os.path.join("saved_models", "ntire", model_name+'.pth')
    if os.path.exists(saved_model_dir):
        print('==> Start testing, current model name: ' + model_name)
        network.load_state_dict(single(saved_model_dir, device))
    
    for idx, batch in enumerate(test_loader):
        img, y = batch
        print("start " + y[0])
        # print(y)
        get_result(img, y[0], device=device, model=network)
        # break

if __name__ == '__main__':
    main()
