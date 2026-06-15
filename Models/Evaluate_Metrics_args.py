"""
Disclaimer: This code is mostly adopted from VEHiCLE:
https://www.nature.com/articles/s41598-021-88115-9
"""
import os, sys, shutil, gzip, argparse, math, time
import time
import multiprocessing
from math import exp

import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from math import log10
import torch.nn as nn
import torch.nn.functional as F
import torch
from Utils.SSIM import ssim
from Utils.GenomeDISCO import compute_reproducibility
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from skimage.metrics import mean_squared_error

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def _toimg(self, mat):
        m = torch.tensor(mat)
        # convert to float and add channel dimension
        return m.float().unsqueeze(0)

    def _tohic(self, mat):
        mat.squeeze_()
        return mat.numpy()  # .astype(int)

    def gaussian(self, width, sigma):
        gauss = torch.Tensor([exp(-(x - width // 2) ** 2 / float(2 * sigma ** 2)) for x in range(width)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel, sigma=3):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def gaussian_filter(self, img, width, sigma=3):
        img = self._toimg(img).unsqueeze(0)
        _, channel, _, _ = img.size()
        window = self.create_window(width, channel, sigma)
        mu1 = F.conv2d(img, window, padding=width // 2, groups=channel)
        return self._tohic(mu1)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def ssim(self, img1, img2, window_size=11, size_average=True):
        img1 = self._toimg(img1).unsqueeze(0)
        img2 = self._toimg(img2).unsqueeze(0)
        _, channel, _, _ = img1.size()
        window = self.create_window(window_size, channel)
        window = window.type_as(img1)

        return self._ssim(img1, img2, window, window_size, channel, size_average)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


class VisionMetrics:
    def __init__(self):
        self.ssim = SSIM()
        self.metric_logs = {
            "pre_psnr": [],
            "pas_psnr": [],
            "pre_snr": [],
            "pas_snr": [],
            "pre_spc": [],
            "pas_spc": [],
            "pre_pcc": [],
            "pas_pcc": [],
            "pre_gds": [],
            "pas_gds": [],
            "pre_ssim": [],
            "pas_ssim": [],
            "pre_mse": [],
            "pas_mse": [],
        }

    def _logSSIM(self, data, target, output):
        self.metric_logs['pre_ssim'].append(self.compareSSIM(data, target))
        self.metric_logs['pas_ssim'].append(self.compareSSIM(output, target))

    def _logPSNR(self, data, target, output):
        self.metric_logs['pre_psnr'].append(self.comparePSNR(data, target))
        self.metric_logs['pas_psnr'].append(self.comparePSNR(output, target))

    def _logPCC(self, data, target, output):
        self.metric_logs['pre_pcc'].append(self.comparePCC(data, target))
        self.metric_logs['pas_pcc'].append(self.comparePCC(output, target))

    def _logSPC(self, data, target, output):
        self.metric_logs['pre_spc'].append(self.compareSPC(data, target))
        self.metric_logs['pas_spc'].append(self.compareSPC(output, target))

    def _logMSE(self, data, target, output):
        self.metric_logs['pre_mse'].append(self.compareMSE(data, target))
        self.metric_logs['pas_mse'].append(self.compareMSE(output, target))

    def _logSNR(self, data, target, output):
        self.metric_logs['pre_snr'].append(self.compareSNR(data, target))
        self.metric_logs['pas_snr'].append(self.compareSNR(output, target))

    def _logGDS(self, data, target, output):
        self.metric_logs['pre_gds'].append(self.compareGDS(data, target))
        self.metric_logs['pas_gds'].append(self.compareGDS(output, target))

    def compareGDS(self, a, b):
        return compute_reproducibility(a[0][0], b[0][0], transition=True)

    def compareSPC(self, a, b):
        return spearmanr(a[0][0], b[0][0], axis=None)[0]

    def comparePCC(self, a, b):
        return pearsonr(a[0][0].flatten(), b[0][0].flatten())[0]

    def comparePSNR(self, a, b):
        MSE = np.square(a[0][0] - b[0][0]).mean().item()
        MAX = torch.max(b).item()
        # print(a.shape, b.shape)
        return 20 * np.log10(MAX) - 10 * np.log10(MSE)

    def compareSNR(self, a, b):
        return torch.sum(b[0][0]).item() / (torch.sqrt(torch.sum((b[0][0] - a[0][0]) ** 2)).item())

    def compareSSIM(self, a, b):
        return self.ssim(a, b).item()

    def compareMSE(self, a, b):
        return np.square(a[0][0] - b[0][0]).mean().item()

    def log_means(self, name):
        return (name, np.mean(self.metric_logs[name]), np.std(self.metric_logs[name]))

    def setDataset(self, model_output, model_input, target):
        self.model_output = model_output
        self.target = target
        self.model_input = model_input

    def getMetrics(self):
        self.metric_logs = {
            "pre_psnr": [],
            "pas_psnr": [],
            "pre_snr": [],
            "pas_snr": [],
            "pre_spc": [],
            "pas_spc": [],
            "pre_pcc": [],
            "pas_pcc": [],
            "pre_gds": [],
            "pas_gds": [],
            "pre_ssim": [],
            "pas_ssim": [],
            "pre_mse": [],
            "pas_mse": [],
        }

        for e, pred in enumerate(self.model_output):


            self._logPCC(data=self.model_input[e:e+1], target=self.target[e:e+1], output=self.model_output[e:e+1])
            self._logSPC(data=self.model_input[e:e+1], target=self.target[e:e+1], output=self.model_output[e:e+1])
            self._logMSE(data=self.model_input[e:e+1], target=self.target[e:e+1], output=self.model_output[e:e+1])
            self._logPSNR(data=self.model_input[e:e+1], target=self.target[e:e+1], output=self.model_output[e:e+1])
            self._logSNR(data=self.model_input[e:e+1], target=self.target[e:e+1], output=self.model_output[e:e+1])
            self._logSSIM(data=self.model_input[e:e+1], target=self.target[e:e+1], output=self.model_output[e:e+1])
            self._logGDS(data=self.model_input[e:e+1], target=self.target[e:e+1], output=self.model_output[e:e+1])
        # print(list(map(self.log_means, self.metric_logs.keys())))
        return list(filter(lambda x: "pre" not in x[0],
                            map(self.log_means, self.metric_logs.keys())))
    
    
def metrics_predictor(lr_data, pred_data, hr_data):

    visionMetrics = VisionMetrics()
    visionMetrics.setDataset(torch.from_numpy(pred_data),
                            torch.from_numpy(lr_data),
                            torch.from_numpy(hr_data))
    
    results = visionMetrics.getMetrics()
    
    
    # hicarn_hics = together(result_data, result_inds, tag='Reconstructing: ')
    # return hicarn_hics
    return results           	
    

# def save_data(carn, compact, size, file):
#     hicarn = spreadM(carn, compact, size, convert_int=False, verbose=True)
#     np.savez_compressed(file, hicarn=hicarn, compact=compact)
#     print('Saving file:', file)


def preprocess_data_raw(test_data, cutoff):
    # test_data = np.moveaxis(test_data, -1, 1)
    max_test_data = np.max(test_data)
    test_data = test_data/max_test_data
    return test_data

def preprocess_data(test_data):
    return np.moveaxis(test_data, -1, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    ## Input args
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--cell_line', type=str, required=True)
    parser.add_argument('--RATIO', type=int, required=True)
    parser.add_argument('--pred_dir', type=str, required=True)
    args = parser.parse_args()

    root_dir = args.root_dir #"./data/40_x_40_new"
    cell_line = args.cell_line #"GM12878_raw_rep1"
    RATIO = args.RATIO #16
    pred_dir = args.pred_dir #f"./results/truhic_2213_32_heads_40x40_new_raw_GM12878_rep1_r{RATIO}/out"

    hr_dir = os.path.join(root_dir, cell_line,  f"{RATIO}_ratio", "to_predict")
    lr_dir = os.path.join(root_dir, cell_line, f"{RATIO}_ratio", "to_predict")
    
    # chrs = [4, 14, 16, 20]
    chrs = [18, 19, 20, 21, 22]
    # chrs = [16, 17, 18, 19]

    hr_file_paths = [os.path.join(hr_dir, f"hr_test_chr{chr}.npy") for chr in chrs]
    lr_file_paths = [os.path.join(lr_dir, f"lr_test_chr{chr}_ratio{RATIO}.npy") for chr in chrs]
    pred_file_paths = [os.path.join(pred_dir, f"preds_lr_test_chr{chr}_ratio{RATIO}.npy") for chr in chrs]
    
    # print("Method:", METHOD_NAME)
    device = torch.device('cpu')
    print(f'Using device: {device}')

    for i, chr in enumerate(chrs):
        print(f'Chromosome {chr}:')
        data_hr = np.load(hr_file_paths[i], allow_pickle=True).astype(np.float64)
        data_lr = np.load(lr_file_paths[i], allow_pickle=True).astype(np.float64)
        data_pred = np.load(pred_file_paths[i], allow_pickle=True).astype(np.float64)
        data_hr = np.moveaxis(data_hr, -1, 1)
        data_lr = np.moveaxis(data_lr, -1, 1)
        data_pred = np.moveaxis(data_pred, -1, 1)

        
        # hicarn_data_hr = preprocess_data_raw(hicarn_data_hr, None)
        # hicarn_data_pred = preprocess_data(hicarn_data_pred)
        print(f"Hr shape: {data_hr.shape}")
        print(f"Pred shape: {data_pred.shape}")

        metrics = metrics_predictor(data_lr, data_pred, data_hr)
        # for metric in metrics:
        #     print(f"Metric: {metric[0]+' ' if len(metric[0]) < 8 else metric[0]}\tmean: {metric[1]:4f}\tstd: {metric[2]:4f}")
        print(f"Avg. over samples for chr {chr}:")
        print("\t".join([str(x[1]) for x in metrics]))
        print("============================================================")
