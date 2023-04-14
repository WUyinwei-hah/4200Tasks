# 4200Tasks

## 分工
每个人分工是对称的，主要需要补充一些实验数据。
1. 从[paper with code](https://paperswithcode.com/task/image-dehazing)选择一个有开源代码的dehazing模型。
2. 使用这个模型在我提供的低分辨率(384, 256)数据集和切patch(256, 256)数据集上跑训练和验证。
3. 对于低分辨率数据集，低分辨率需要提供所有验证集图片(384, 256)的PSNR，SSIM以及验证集生成的图片。
4. 对于patch数据集，需要提供(6000, 4000)测试集图片的PSNR以及SSIM。还有拼接完成后的测试集图片。

## 数据集下载地址：
1. [低分数据集](https://drive.google.com/file/d/1sheHZJXVY9swZMpDGl8ZFcYzNdzKRLn4/view?usp=sharing)，这个数据集是经过扩充的有200+张，包括训练验证，测试（没有GT）
2. [原数据集](https://drive.google.com/file/d/1e8mvPlNMm2A1rpqzNvNPuswVMFPCU7gK/view?usp=share_link)，可以随意切patch使用，没有扩充，只有40张

## 关于本repo
本repo，提供了一些模板代码，可以参考使用：
1. PSNR，SSIM计算工具（metrics.py）
2. patch合并成高分图片的工具（concat_2_hr.py）
3. dataloader（dataloader.py）
4. 切patch工具（crop.py）
