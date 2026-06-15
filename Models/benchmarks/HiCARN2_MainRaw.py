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
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from Models.HiCARN_2 import Generator, Discriminator
from Models.HiCARN_2_Loss import GeneratorLoss
from Utils.SSIM import ssim
from math import log10

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
    # Normalize outliers
    # hr_train_cutoff = np.percentile(hr_train, 99.9)
    # # hr_valid_cutoff = np.percentile(hr_valid, 99.9)
    # hr_train = np.minimum(hr_train, hr_train_cutoff)
    # hr_valid = np.minimum(hr_valid, hr_train_cutoff)
    # lr_train = np.minimum(lr_train, hr_train_cutoff)
    # lr_valid = np.minimum(lr_valid, hr_train_cutoff)

    # #Scale values
    # max_hr_train = np.max(hr_train)
    # # max_hr_valid = np.max(hr_valid)
    # hr_train = hr_train/max_hr_train
    # lr_train = lr_train/max_hr_train
    # hr_valid = hr_valid/max_hr_train
    # lr_valid = lr_valid/max_hr_train

    return lr_train, hr_train, lr_valid, hr_valid


def load_test_data(input_dir, ratio):
    if "to_predict" not in input_dir:
        lr_test = np.load(f"{input_dir}/{ratio}_ratio/lr_test_ratio{ratio}.npy").astype("float32")
        lr_test = np.moveaxis(lr_test, -1, 1)
        #Scale values
        # scale_val = np.max(lr_test)
        # lr_test /= scale_val
        return {f"lr_test_ratio{ratio}":(lr_test, 1.0)}
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
            outputs[file_name.split(".")[0]] = (lr_test, 1.0)
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
    netG = Generator(num_channels=64).to(device)
    netD = Discriminator().to(device)

    # loss function
    criterionG = GeneratorLoss().to(device)
    criterionD = torch.nn.BCELoss().to(device)

    # optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=0.0003)
    optimizerD = optim.Adam(netD.parameters(), lr=0.0003)

    ssim_scores = []
    psnr_scores = []
    mse_scores = []
    mae_scores = []

    best_ssim = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        run_result = {'nsamples': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
        loss_mem = []
        alr = adjust_learning_rate(epoch)
        optimizerG = optim.Adam(netG.parameters(), lr=alr)
        optimizerD = optim.Adam(netD.parameters(), lr=alr)

        for p in netG.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        torch.cuda.empty_cache()
        start_time = time.time()
        netG.train()
        netD.train()
        # train_bar = tqdm(train_loader)
        for data, target in train_loader:
            batch_size = data.size(0)
            run_result['nsamples'] += batch_size
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = target.to(device)
            z = data.to(device)
            fake_img = netG(z)

            ######### Train discriminator #########
            netD.zero_grad()
            real_out = netD(real_img)
            fake_out = netD(fake_img)
            d_loss_real = criterionD(real_out, torch.ones_like(real_out))
            d_loss_fake = criterionD(fake_out, torch.zeros_like(fake_out))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ######### Train generator #########
            netG.zero_grad()
            g_loss = criterionG(fake_out.mean(), fake_img, real_img)
            g_loss.backward()
            optimizerG.step()

            run_result['g_loss'] += g_loss.item() * batch_size
            run_result['d_loss'] += d_loss.item() * batch_size
            run_result['d_score'] += real_out.mean().item() * batch_size
            run_result['g_score'] += fake_out.mean().item() * batch_size

            # train_bar.set_description(
            #     desc=f"[{epoch}/{NUM_EPOCHS}] "
            #         f"Loss_D: {run_result['d_loss'] / run_result['nsamples']:.4f} "
            #         f"Loss_G: {run_result['g_loss'] / run_result['nsamples']:.4f} "
            #         f"D(x): {run_result['d_score'] / run_result['nsamples']:.4f} "
            #         f"D(G(z)): {run_result['g_score'] / run_result['nsamples']:.4f}")
            loss_mem.append(run_result['g_loss'] / run_result['nsamples'])
        
        end_time = time.time()
        print(f"Average training loss for epoch {epoch}/{NUM_EPOCHS}: {np.mean(loss_mem):.5f} [Took {(end_time-start_time):.5f} seconds]")
        train_gloss = run_result['g_loss'] / run_result['nsamples']
        train_dloss = run_result['d_loss'] / run_result['nsamples']
        train_dscore = run_result['d_score'] / run_result['nsamples']
        train_gscore = run_result['g_score'] / run_result['nsamples']

        valid_result = {'g_loss': 0, 'd_loss': 0, 'g_score': 0, 'd_score': 0,
                        'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
        netG.eval()
        netD.eval()

        batch_ssims = []
        batch_mses = []
        batch_psnrs = []
        batch_maes = []

        # valid_bar = tqdm(valid_loader)
        loss_mem = []
        start_time = time.time()
        with torch.no_grad():
            
            for val_lr, val_hr in valid_loader:
                batch_size = val_lr.size(0)
                valid_result['nsamples'] += batch_size
                lr = val_lr.to(device)
                hr = val_hr.to(device)
                sr = netG(lr)

                sr_out = netD(sr)
                hr_out = netD(hr)
                d_loss_real = criterionD(hr_out, torch.ones_like(hr_out))
                d_loss_fake = criterionD(sr_out, torch.zeros_like(sr_out))
                d_loss = d_loss_real + d_loss_fake
                g_loss = criterionG(sr_out.mean(), sr, hr)

                valid_result['g_loss'] += g_loss.item() * batch_size
                valid_result['d_loss'] += d_loss.item() * batch_size
                valid_result['g_score'] += sr_out.mean().item() * batch_size
                valid_result['d_score'] += hr_out.mean().item() * batch_size

                batch_mse = ((sr - hr) ** 2).mean()
                batch_mae = (abs(sr - hr)).mean()
                valid_result['mse'] += batch_mse * batch_size
                batch_ssim = ssim(sr, hr)
                valid_result['ssims'] += batch_ssim * batch_size
                valid_result['psnr'] = 10 * log10(1 / (valid_result['mse'] / valid_result['nsamples']))
                valid_result['ssim'] = valid_result['ssims'] / valid_result['nsamples']
                # valid_bar.set_description(
                #     desc=f"[Predicting in Test set] PSNR: {valid_result['psnr']:.4f} dB SSIM: {valid_result['ssim']:.4f}")
                loss_mem.append(run_result['g_loss'] / run_result['nsamples'])
                # print(f"[Predicting in Test set] PSNR: {valid_result['psnr']:.4f} dB SSIM: {valid_result['ssim']:.4f}")
                batch_ssims.append(valid_result['ssim'])
                batch_psnrs.append(valid_result['psnr'])
                batch_mses.append(batch_mse)
                batch_maes.append(batch_mae)

        end_time = time.time()
        print(f"Average validation loss for epoch {epoch}/{NUM_EPOCHS}: {np.mean(loss_mem):.5f} [Took {(end_time-start_time):.5f} seconds]")
        ssim_scores.append((sum(batch_ssims) / len(batch_ssims)))
        psnr_scores.append((sum(batch_psnrs) / len(batch_psnrs)))
        mse_scores.append((sum(batch_mses) / len(batch_mses)))
        mae_scores.append((sum(batch_maes) / len(batch_maes)))

        valid_gloss = valid_result['g_loss'] / valid_result['nsamples']
        valid_dloss = valid_result['d_loss'] / valid_result['nsamples']
        valid_gscore = valid_result['g_score'] / valid_result['nsamples']
        valid_dscore = valid_result['d_score'] / valid_result['nsamples']
        now_ssim = valid_result['ssim'].item()


        if now_ssim > best_ssim:
            best_ssim = now_ssim
            print(f'Now, Best ssim is {best_ssim:.6f}')
            best_ckpt_file = f'best_model.pytorch'
            torch.save(netG.state_dict(), os.path.join(args.save_dir, "models", best_ckpt_file))
    pass


def impute_the_target(args):
    if os.path.exists(f"{args.save_dir}/commandline_args.json"):
        with open(f"{args.save_dir}/commandline_args.json", 'r') as f:
            training_args = json.load(f)
        # Ensure that the same cut-off of the training set is used for dealing with outlier reads

    model_path = os.path.join(args.save_dir, "models", f'best_model.pytorch')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load network
    netG = Generator(num_channels=64).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    netG.eval()
    BATCH_SIZE = args.batch_size_per_gpu 

    with torch.no_grad():
        lr_test_dict = load_test_data(args.input, args.ratio)
        for keey, (vall, scale_val) in lr_test_dict.items():
            result_data = []
            print(f"Predicting {keey} with shape of {vall.shape}")
            test_loader = dataloader(vall, hr_data=None, batch_size=BATCH_SIZE)
            for lr in tqdm(test_loader, desc='HiCARN Predicting: '):
                lr = lr[0].to(device)
                out = netG(lr).cpu().detach().numpy()
                
                result_data.append(out)
            result_data = np.concatenate(result_data, axis=0)
            result_data = np.moveaxis(result_data, 1, -1).astype(np.float32) * scale_val
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
    deciding_args_parser = argparse.ArgumentParser(description='HiCARN1.', add_help=False)

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
    parser.add_argument('--epochs', type=int, required=False, help='Maximum number of epochs (default 100)',
                        default=100)
    parser.add_argument('--lr', type=float, required=False, help='Learning Rate (default 0.0003)', default=0.0003)
    parser.add_argument('--batch-size-per-gpu', type=int, required=False, help='Batch size per gpu(default 16)',
                        default=64)

    args = parser.parse_args()
    args.restart_training = str_to_bool(args.restart_training)

    if not (args.save_dir.startswith("./") or args.save_dir.startswith("/")):
        args.save_dir = f"./{args.save_dir}"
    print(f"Save directory will be:\t{args.save_dir}")

    if args.mode == 'train':
        train_the_model(args)
    else:
        impute_the_target(args)


if __name__ == '__main__':
    main()
