import glob
import sys
from collections import OrderedDict

from natsort import natsort
import argparse
import options.options as option
from Measure import Measure, psnr
from imresize import imresize
from models import create_model
import torch
from utils.util import opt_get
import numpy as np
import pandas as pd
import os
import cv2
from utils import util

def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt


def predict(model, lr):
    model.feed_data({"LQ": t(lr)}, need_GT=False)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    return visuals.get('rlt', visuals.get('NORMAL'))


def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def imCropCenter(img, size):
    h, w, c = img.shape

    h_start = max(h // 2 - size // 2, 0)
    h_end = min(h_start + size, h)

    w_start = max(w // 2 - size // 2, 0)
    w_end = min(w_start + size, w)

    return img[h_start:h_end, w_start:w_end]


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')


def impad2(input,divide):
    height_org, width_org = input.shape[2], input.shape[3]

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = torch.nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[1], input.shape[2]
    return input[ :, pad_top: height - pad_bottom, pad_left: width - pad_right]


def hiseq_color_cv2_img(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", default=".\confs/LOLv2-pc.yml")
    args = parser.parse_args()
    conf_path = args.opt
    conf = conf_path.split('/')[-1].replace('.yml', '')
    model, opt = load_model(conf_path)
    model.netG = model.netG.cuda()

    lr_dir = opt['dataroot_LR']
    hr_dir = opt['dataroot_GT']

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
    hr_paths = fiFindByWildcard(os.path.join(hr_dir, '*.png'))

    this_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(this_dir, '..', 'results', conf)
    print(f"Out dir: {test_dir}")

    measure = Measure(use_gpu=False)

    df = None

    scale = opt['scale']

    pad_factor = 8

    for lr_path, hr_path, idx_test in zip(lr_paths, hr_paths, range(len(lr_paths))):

        lr = imread(lr_path)
        hr = imread(hr_path)
        # Pad image to be % 2
        h, w, c = lr.shape
        if h % pad_factor != 0 or w % pad_factor != 0:
            lr = impad(lr, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                       right=int(np.ceil(w / pad_factor) * pad_factor - w))

        his = hiseq_color_cv2_img(lr)
        if opt.get("histeq_as_input", False):
            lr = his

        lr_t = t(lr)
        zero = make_zero_img(lr_t)
        if opt["datasets"]["train"].get("log_low", False):
            lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
        if opt.get("concat_histeq", False):
            his = t(his)
            lr_t = torch.cat([lr_t, his], dim=1)
        heat = 1

        # if df is not None and len(df[(df['heat'] == heat) & (df['name'] == idx_test)]) == 1:
        #     continue
        with torch.cuda.amp.autocast():  ##这里进入网络
            sr_t = model.get_sr(lq=lr_t.cuda(),zero_channel=zero.cuda(), heat=None)

        sr = rgb(torch.clamp(sr_t, 0, 1))
        sr = sr[:h * scale, :w * scale]
        path_out_sr = os.path.join(test_dir, "{:0.2f}".format(heat).replace('.', ''), os.path.basename(hr_path))
        imwrite(path_out_sr, sr)

        meas = OrderedDict(conf=conf, heat=heat, name=idx_test)
        meas['PSNR'], meas['SSIM'], meas['LPIPS'] = measure.measure(sr, hr)

        str_out = format_measurements(meas)
        print(str_out)

        df = pd.DataFrame([meas]) if df is None else pd.concat([pd.DataFrame([meas]), df])

    str_out = format_measurements(df.mean())
    print('Mean: ' + str_out)

def make_zero_img(input_img):
    dark = input_img  # 输入暗图像
    R_split, G_split, B_split = torch.split(dark, 1, dim=1)
    zero_array = R_split * G_split * B_split
    zero_array[zero_array != 0] = 1
    zero_array = 1 - zero_array
    mask = zero_array
    return mask
def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.6f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out


if __name__ == "__main__":
    main()
