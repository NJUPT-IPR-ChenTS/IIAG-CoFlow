# IIAG-CoFlow: Inter-and Intra-channel Attention Transformer and Complete Flow for Low-light Image Enhancement
## Introduction
IIAG-CoFlow is a novel normalizing ﬂow learning
 based method IIAG-CoFlow for low-light image enhancement
(LLIE), which consists of an inter-and intra-channel attention
Transformer based conditional generator (IIAG) and a complete
ﬂow (CoFlow). **Experiments show
that IIAG-CoFlow outperforms existing SOTA LLIE methods
on several benchmark low-light datasets.**
## Evaluation Metrics
| Methods | LOL-v2-real |LOL-v2-synthetic|SMID |SDSD-indoor|SDSD-outdoor|SID|
|:--------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|/|**PSNR/SSIM/LPIPS**|**PSNR/SSIM/LPIPS**|**PSNR/SSIM/LPIPS**|**PSNR/SSIM/LPIPS**|**PSNR/SSIM/LPIPS**|**PSNR/SSIM/LPIPS**|
|LLFlow|19.67/0.852/0.157|22.38/0.910/0.066|28.12/0.813/0.181|25.46/0.896/0.139|28.82/0.869/0.142|19.39/0.615/0.386|
|LLFormer|20.99/0.801/0.219|23.74/0.902/0.086|27.92/0.785/0.183|29.65/0.874/0.152|28.73/0.838/0.129|21.26/0.575/0.481|
|MIRNet|21.18/0.840/0.145|25.08/0.920/0.070|28.67/0.810/0.180|27.83/0.882/0.138|29.17/0.871/0.152|20.87/0.605/0.460|
|SNR|21.48/0.849/0.157|24.14/0.928/0.056|28.49/0.805/0.178|29.44/0.894/0.129|28.66/0.866/0.140|22.87/0.619/0.359|
|Retinexformer|22.79/0.839/0.171|25.67/0.928/0.059|28.99/0.811/0.167|29.75/0.892/0.118|29.83/0.875/0.178|**24.44/0.675/0.344**|
|SMG|24.03/0.820/0.169|24.98/0.894/0.092|26.97/0.725/0.211|26.89/0.802/0.166|26.33/0.809/**0.093**|22.63/0.541/0.377|
|**IIAG-CoFlow(Ours)**|**25.16/0.893/0.108**|**26.57/0.946/0.038**|**29.31/0.822/0.160**|**31.59/0.913/0.106**|**31.07/0.893**/0.164|22.31/0.656/0.374|

(We remove the GT correction operation when obtaining the metrics of LLFlow for fair comparison. The enhanced image of SMG is with the size 512×512×3 that is different from the original ground truth (GT) image of each testing set, we rescale the enhanced images of SMG to have the same size with the original GT image of each testing set for fair comparison.)
## Visual Quality
* Visual comparison of different methods on LOL-v2-real
<div align="center">
  <img src="https://github.com/NJUPT-IPR-ChenTS/pic/blob/main/fig7.jpg">
</div>

* Visual comparison of different methods on LOL-v2-synthetic
<div align="center">
  <img src="https://github.com/NJUPT-IPR-ChenTS/pic/blob/main/fig8.jpg">
</div>

* Visual comparison of different methods on SID
<div align="center">
  <img src="https://github.com/NJUPT-IPR-ChenTS/pic/blob/main/fig9.jpg">
</div>

* Visual comparison of different methods on SMID
<div align="center">
  <img src="https://github.com/NJUPT-IPR-ChenTS/pic/blob/main/fig10.jpg">
</div>

* Visual comparison of different methods on SDSD-outdoor
<div align="center">
  <img src="https://github.com/NJUPT-IPR-ChenTS/pic/blob/main/fig11.jpg">
</div>

## Dataset
* LOLv2 (Real & Synthetic): Please refer to the paper [From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement (CVPR 2020)](https://github.com/flyywh/CVPR-2020-Semi-Low-Light).

* SID & SMID & SDSD (indoor & outdoor): Please refer to the paper
[SNR-aware Low-Light Image Enhancement (CVPR 2022)](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance).

## Testing
### Pre-trained Models
Please download our pre-trained models via the following links 
[baiduyun(extracted code): 33ee](https://pan.baidu.com/s/1Ke310o5Rv2MP_fDNWb96Hg?pwd=33ee)
### Run the testing code
You can test the model with paired data and obtain the evaluation metrics. You need to specify the data path `dataroot_LR`, `dataroot_GT`, and `model path` model_path in the config file. Then run

```c
python test.py
```

## Acknowledgments
Our code is based on [LLFlow](https://github.com/wyf0912/LLFlow) 

## Contact
If you have any questions, please feel free to contact the authors via
[chentiesheng147@163.com](chentiesheng147@163.com)
