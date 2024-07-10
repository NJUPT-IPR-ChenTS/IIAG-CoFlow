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
