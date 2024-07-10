
#
# import torch
# import torch.nn as nn
# #from models.modules.network_enswinr_res_02 import ENSwinIR
# #from models.modules.restormer_arch import Restormer
# from models.modules.zero_transformer import Restormer
#
#
# #####重要
# class ConEncoder1(nn.Module):
#     def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, opt=None):
#         self.opt = opt
#         self.gray_map_bool = False
#         self.concat_color_map = False
#         if opt['concat_histeq']:
#             in_nc = in_nc + 3
#         if opt['concat_color_map']:
#             in_nc = in_nc + 3
#             self.concat_color_map = True
#         if opt['gray_map']:
#             in_nc = in_nc + 1
#             self.gray_map_bool = True
#
#         super(ConEncoder1, self).__init__()
#
#         self.scale = scale
#
#         #self.MTA=ENSwinIR()
#         self.Restormer=Restormer()
#
# #######重要
#     def forward(self, x,zc):#x是低光照和直方图均衡化拼在一起，6通道，192*192
#         if self.gray_map_bool:
#             x = torch.cat([x, 1 - x.mean(dim=1, keepdim=True)], dim=1)
#         if self.concat_color_map:
#             x = torch.cat([x, x / (x.sum(dim=1, keepdim=True) + 1e-4)], dim=1)
#
#         # raw_low_input = x[:, 0:3].exp()#对低光照通道处理
#         # # fea_for_awb = F.adaptive_avg_pool2d(fea_down8, 1).view(-1, 64)
#         # awb_weight = 1  # (1 + self.awb_para(fea_for_awb).unsqueeze(2).unsqueeze(3))
#         # low_after_awb = raw_low_input * awb_weight
#         # # import pdb
#         # # pdb.set_trace()
#         # color_map = low_after_awb / (low_after_awb.sum(dim=1, keepdims=True) + 1e-4)#计算颜色图
#         # dx, dy = self.gradient(color_map)
#         # noise_map = torch.max(torch.stack([dx.abs(), dy.abs()], dim=0), dim=0)[0]   ###噪声图，
#         # # color_map = self.fine_tune_color_map(torch.cat([color_map, noise_map], dim=1))
#         # x=torch.cat([x, color_map, noise_map], dim=1)
#
#         results={}
#         results=self.Restormer(x,zc,results)
#
#         return results
#
# #####有用
#     def gradient(self, x):
#         def sub_gradient(x):
#             left_shift_x, right_shift_x, grad = torch.zeros_like(
#                 x), torch.zeros_like(x), torch.zeros_like(x)
#             left_shift_x[:, :, 0:-1] = x[:, :, 1:]
#             right_shift_x[:, :, 1:] = x[:, :, 0:-1]
#             grad = 0.5 * (left_shift_x - right_shift_x)
#             return grad
#
#         return sub_gradient(x), sub_gradient(torch.transpose(x, 2, 3)).transpose(2, 3)
#

