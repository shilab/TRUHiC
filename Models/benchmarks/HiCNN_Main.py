import os, sys, shutil, gzip, argparse, math, time
import gzip
from tqdm import tqdm
import numpy as np
from typing import Union
from scipy.special import softmax
from scipy.spatial.distance import squareform
import json
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from Models.HiCNN import Generator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Utils.SSIM import ssim
from math import log10
import random

random.seed(0)
torch.manual_seed(0)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def pprint(text):
    print(f"{bcolors.OKGREEN}{text}{bcolors.ENDC}")


def adjust_learning_rate(epoch):
    lr = 0.0003 * (0.1 ** (epoch // 30))
    return lr

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']
    
def create_directories(save_dir,
                       models_dir="models",
                       outputs="out") -> None:
    for dd in [save_dir,
               f"{save_dir}/{models_dir}",
               f"{save_dir}/{outputs}"]:
        if not os.path.exists(dd):
            os.makedirs(dd)
    pass


def clear_dir(path) -> None:
    # credit: https://stackoverflow.com/a/72982576/4260559
    if os.path.exists(path):
        for entry in os.scandir(path):
            if entry.is_dir():
                clear_dir(entry)
            else:
                os.remove(entry)
        os.rmdir(path)  # if you just want to delete the dir content but not the dir itself, remove this line


def map_pos_matrics_to_values(val_tuple):
    return [int(val_tuple[1][3:])]

def load_training_data(input_dir, ratio):
    # hr_valid = np.load(f"{input_dir}/{ratio}_ratio/hr_valid.npy").astype("float32")
    # lr_valid = np.load(f"{input_dir}/{ratio}_ratio/lr_valid_ratio{ratio}.npy").astype("float32")

    hr_train = np.load(f"{input_dir}/{ratio}_ratio/hr_train.npy").astype("float32")
    lr_train = np.load(f"{input_dir}/{ratio}_ratio/lr_train_ratio{ratio}.npy").astype("float32")

    hr_train = np.moveaxis(hr_train, -1, 1)
    lr_train = np.moveaxis(lr_train, -1, 1)

    chr_numbers = np.load(f"{input_dir}/{ratio}_ratio/distance_train.npy")
    chr_numbers = np.apply_along_axis(map_pos_matrics_to_values, -1, chr_numbers).astype("int32")

    train_indices, valid_indices = train_test_split(np.arange(len(hr_train)),
                                     test_size=.1,
                                     random_state=2025,
                                     shuffle=True,
                                     stratify=chr_numbers
                                     )
    
    hr_valid, lr_valid = hr_train[valid_indices], lr_train[valid_indices]
    hr_train, lr_train = hr_train[train_indices], lr_train[train_indices]

    return lr_train, hr_train, lr_valid, hr_valid

def load_test_data(input_dir, ratio):
    if "to_predict" not in input_dir:
        lr_test = np.load(f"{input_dir}/{ratio}_ratio/lr_test_ratio{ratio}.npy").astype("float32")
        lr_test = np.moveaxis(lr_test, -1, 1)
        #Scale values
        # scale_val = np.max(lr_test)
        # lr_test /= scale_val
        return {f"lr_test_ratio{ratio}":lr_test}
    else:
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and "hr_" not in f]
        outputs = {}
        for file_name in files:
            lr_test = np.load(os.path.join(input_dir, file_name)).astype("float32")
            print("File lr shape: ", lr_test.shape)
            lr_test = np.moveaxis(lr_test, -1, 1)

            #Scale values
            # scale_val = np.max(lr_test)
            # lr_test /= scale_val
            outputs[file_name.split(".")[0]] = lr_test
        return outputs
    
cs = np.column_stack


def adjust_learning_rate(epoch):
    lr = 0.0003 * (0.1 ** (epoch // 30))
    return lr


def dataloader(lr_data, hr_data=None, batch_size=64):
    inputs = torch.tensor(lr_data, dtype=torch.float)
    if hr_data is not None:
        target = torch.tensor(hr_data, dtype=torch.float)
        dataset = TensorDataset(inputs, target)
    else:
        dataset = TensorDataset(inputs)
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def train_the_model(args) -> None:
    if args.restart_training:
        clear_dir(args.save_dir)

    NUM_EPOCHS = args.epochs
    
    BATCH_SIZE = args.batch_size_per_gpu

    create_directories(args.save_dir)
    
    lr_train, hr_train, lr_valid, hr_valid = load_training_data(args.input, args.ratio)

    with open(f"{args.save_dir}/commandline_args.json", 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("CUDA available? ", torch.cuda.is_available())
    print("Device being used: ", device)

    train_data = torch.tensor(lr_train, dtype=torch.float)
    train_target = torch.tensor(hr_train, dtype=torch.float)
    train_set = TensorDataset(train_data, train_target)


    valid_data = torch.tensor(lr_valid, dtype=torch.float)
    valid_target = torch.tensor(hr_valid, dtype=torch.float)
    valid_set = TensorDataset(valid_data, valid_target)

    # DataLoader for batched training
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)


    # load network
    netG = Generator(input_channels=1).to(device)
    # loss function
    criterionG = nn.MSELoss().to(device)

    # optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=0.0003)
    # scheduler = ReduceLROnPlateau(optimizerG, 'min')
    ssim_scores = []
    psnr_scores = []
    mse_scores = []
    mae_scores = []
    best_ssim = 0
    best_vloss = 999999
    for epoch in range(1, NUM_EPOCHS + 1):
        run_result = {'nsamples': 0, 'g_loss': 0, 'g_score': 0}
        alr = adjust_learning_rate(epoch)
        optimizerG = optim.Adam(netG.parameters(), lr=alr)
        for p in netG.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        torch.cuda.empty_cache()

        netG.train()
        # train_bar = tqdm(train_loader)
        step = 0
        for data, target in train_loader:
            # data = data[:,:1,:,:]
            # target = target[:,:1,:,:]
            step += 1
            batch_size = data.size(0)
            run_result['nsamples'] += batch_size

            real_img = target.to(device)
            z = F.pad(data, (6, 6, 6, 6), mode='constant')
            z = z.to(device)
            fake_img = netG(z)

            ######### Train generator #########
            netG.zero_grad()
            g_loss = criterionG(fake_img, real_img)
            g_loss.backward()
            optimizerG.step()

            run_result['g_loss'] += g_loss.item() * batch_size
            # train_bar.set_description(
            #     desc=f"[{epoch}/{NUM_EPOCHS}] Loss_G: {run_result['g_loss'] / run_result['nsamples']:.4f}")
        train_gloss = run_result['g_loss'] / run_result['nsamples']

        valid_result = {'g_loss': 0,
                        'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
        netG.eval()

        batch_ssims = []
        batch_mses = []
        batch_psnrs = []
        batch_maes = []

        # valid_bar = tqdm(valid_loader)
        with torch.no_grad():
            for val_lr, val_hr in valid_loader:
                # val_lr = val_lr[:,:1,:,:]
                # val_hr = val_hr[:,:1,:,:]
                batch_size = val_lr.size(0)
                valid_result['nsamples'] += batch_size
                lr = F.pad(val_lr, (6, 6, 6, 6), mode='constant')
                lr = lr.to(device)
                hr = val_hr.to(device)
                sr = netG(lr)

                sr_out = sr
                hr_out = hr
                g_loss = criterionG(sr, hr)

                valid_result['g_loss'] += g_loss.item() * batch_size

                batch_mse = ((sr - hr) ** 2).mean()
                batch_mae = (abs(sr - hr)).mean()
                valid_result['mse'] += batch_mse * batch_size
                batch_ssim = ssim(sr, hr)
                valid_result['ssims'] += batch_ssim * batch_size
                valid_result['psnr'] = 10 * log10(1 / (valid_result['mse'] / valid_result['nsamples']))
                valid_result['ssim'] = valid_result['ssims'] / valid_result['nsamples']
                # valid_bar.set_description(
                #     desc=f"[Predicting in Test set] PSNR: {valid_result['psnr']:.4f} dB SSIM: {valid_result['ssim']:.4f}")

                batch_ssims.append(valid_result['ssim'])
                batch_psnrs.append(valid_result['psnr'])
                batch_mses.append(batch_mse)
                batch_maes.append(batch_mae)
        ssim_scores.append((sum(batch_ssims) / len(batch_ssims)))
        psnr_scores.append((sum(batch_psnrs) / len(batch_psnrs)))
        mse_scores.append((sum(batch_mses) / len(batch_mses)))
        mae_scores.append((sum(batch_maes) / len(batch_maes)))

        valid_gloss = valid_result['g_loss'] / valid_result['nsamples']

        if valid_result['g_loss'] < best_vloss:
            best_vloss = valid_result['g_loss']
            print(f'Epoch {epoch}: Best vloss is {best_vloss:.6f}')
            best_ckpt_file = f'best_model.pytorch'
            torch.save(netG.state_dict(), os.path.join(args.save_dir, "models", best_ckpt_file)) 

        # lr_current = get_lr(optimizerG)
        # clip2 = 0.01/lr_current
        # nn.utils.clip_grad_norm_(netG.parameters(),clip2)
        # scheduler.step(valid_gloss)
    final_ckpt_g = f'final_model.pytorch'
    torch.save(netG.state_dict(), os.path.join(args.save_dir, "models", final_ckpt_g))
    pass


def impute_the_target(args):
    if os.path.exists(f"{args.save_dir}/commandline_args.json"):
        with open(f"{args.save_dir}/commandline_args.json", 'r') as f:
            training_args = json.load(f)
        # Ensure that the same cut-off of the training set is used for dealing with outlier reads
    model_path = os.path.join(args.save_dir, "models", f'best_model.pytorch')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.save_dir, "models", f'final_model.pytorch')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load network
    netG = Generator(input_channels=1).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    netG.eval()
    BATCH_SIZE = args.batch_size_per_gpu 

    with torch.no_grad():
        lr_test_dict = load_test_data(args.input, args.ratio)
        for keey, vall in lr_test_dict.items():
            result_data = []
            print(f"Predicting {keey} with shape of {vall.shape}")
            test_loader = dataloader(vall, hr_data=None, batch_size=BATCH_SIZE)
            for lr in tqdm(test_loader, desc='HiCNN Predicting: '):
                lr = lr[0].to(device)
                z = F.pad(lr, (6, 6, 6, 6), mode='constant')
                z = z.to(device)
                out = netG(z).cpu().detach().numpy()
                
                result_data.append(out)
            result_data = np.concatenate(result_data, axis=0)
            result_data = np.moveaxis(result_data, 1, -1).astype(np.float32)# * scale_val
            # for ii in range(len(test_preds)):
            #     print(f"Output[{ii}] shape: {test_preds[ii].shape}")
            # with gzip.GzipFile(f"{args.save_dir}/out/test_preds.npy.gz", "w") as f:
            np.save(f"{args.save_dir}/out/preds_{keey}.npy", arr=result_data)
            # np.save(f"{args.save_dir}/out/test_preds", test_preds)
            filesize = os.path.getsize(f"{args.save_dir}/out/preds_{keey}.npy") >> 20
            print(f"The file size is {filesize}MBs")

def str_to_bool(s):
    # Define accepted string values for True and False
    true_values = ['true', '1']
    false_values = ['false', '0']

    # Convert the input string to lowercase for case-insensitive comparison
    lower_s = s.lower()

    # Check if the input string is in the list of true or false values
    if lower_s in true_values:
        return True
    elif lower_s in false_values:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {s}. Accepted values are 'true', 'false', '0', '1'.")


def main():
    deciding_args_parser = argparse.ArgumentParser(description='HiCNN.', add_help=False)

    ## Function mode
    deciding_args_parser.add_argument('--mode', type=str, help='Operation mode: denoise | train (default=train)',
                                      choices=['enhance', 'train'], default='train')
    deciding_args_parser.add_argument('--restart-training', type=str, required=False,
                                      help='Whether to clean previously saved models in target directory and restart the training',
                                      choices=['false', 'true', '0', '1'], default='0')
    deciding_args, _ = deciding_args_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        description="", parents=[deciding_args_parser])
    ## Input args
    parser.add_argument('--input', type=str, required=True, help='Input directory path.')
    parser.add_argument('--ratio', type=str, required=True, help='Downsampling ratio.')
    # parser.add_argument('--target', type=str, required=(deciding_args.mode != 'train'),
    #                     help='Target file path. Must be provided in "enhance" mode.')

    ## save args
    parser.add_argument('--save-dir', type=str, required=True, help='the path to save the results and the model.\n'
                                                                    'This path is also used to load a trained model for enhancement.')
    ## Model (hyper-)params
    parser.add_argument('--epochs', type=int, required=False, help='Maximum number of epochs (default 500)',
                        default=500)
    parser.add_argument('--lr', type=float, required=False, help='Learning Rate (default 0.0003)', default=0.0003)
    parser.add_argument('--batch-size-per-gpu', type=int, required=False, help='Batch size per gpu(default 64)',
                        default=64)

    args = parser.parse_args()
    args.restart_training = str_to_bool(args.restart_training)

    if not (args.save_dir.startswith("./") or args.save_dir.startswith("/")):
        args.save_dir = f"./{args.save_dir}"
    pprint(f"Save directory will be:\t{args.save_dir}")

    if args.mode == 'train':
        train_the_model(args)
    else:
        impute_the_target(args)


if __name__ == '__main__':
    main()
