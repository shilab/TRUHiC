import os, sys, shutil, gzip, argparse, math
import gzip
from tqdm import tqdm
import numpy as np
from typing import Union
from scipy.special import softmax
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import tensorflow_addons as tfa
from scipy.spatial.distance import squareform
import datatable as dt
import json
import os
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv1D, Conv2D, PReLU,LayerNormalization, Flatten, UpSampling2D, LeakyReLU, Dense, Input, add, Lambda
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, BatchNormalization, Activation, MaxPool2D, Concatenate, Add
from DFHiC_TF2 import ModelWrapper, dfhic


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

def find_next_multiple_of_m(number, m):
  ceiling_value = math.ceil(number)
  if ceiling_value % m == 0:
      return ceiling_value
  next_multiple_of_m = ceiling_value + (m - (ceiling_value % m))

  return int(next_multiple_of_m)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# logging.basicConfig(level=logging.WARNING)
pprint("Tensorflow version " + tf.__version__)


keras.saving.get_custom_objects().clear()


custom_objects = {"dfhic": dfhic,
                    "ModelWrapper": ModelWrapper,
                  }


## Model creation
def create_model(args):
    model = ModelWrapper()
                  
    optimizer = tf.keras.optimizers.Adam(args["lr"], beta_1=args["beta1"],
                                        #   weight_decay=lr_decay
                                          )
    model.compile(optimizer, loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])
    return model


def create_callbacks(metric="val_mse", save_path="."):
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=metric,
        mode='auto',
        factor=0.5,
        patience=7,
        min_delta=1e-7,
        verbose=0
    )

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor=metric,
        mode='auto',
        patience=20,
        verbose=1,
        restore_best_weights=True
    )

    callbacks = [reducelr, earlystop]

    return callbacks


def get_dataset(lr_set, batch_size, strategy, hr_set=None, training=False):
    AUTO = tf.data.AUTOTUNE
    if hr_set is not None:
        dataset = tf.data.Dataset.from_tensor_slices((lr_set, hr_set))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((lr_set))

    if training and hr_set is not None:
        dataset = dataset.shuffle(lr_set.shape[0], reshuffle_each_iteration=True)
        dataset = dataset.repeat()
    dataset = dataset.prefetch(AUTO)\
                    .batch(batch_size, drop_remainder=training, num_parallel_calls=AUTO)
    # This part is for multi-gpu servers
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    dataset = dataset.with_options(options)
    dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset



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

    hr_train = np.load(f"{input_dir}/{ratio}_ratio/hr_train.npy").astype("float32")[..., 0:1]
    lr_train = np.load(f"{input_dir}/{ratio}_ratio/lr_train_ratio{ratio}.npy").astype("float32")

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
        if lr_test.shape[1] == 1:
            lr_test = np.moveaxis(lr_test, 1, -1)
        return {f"lr_test_ratio{ratio}":lr_test}
    else:
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and "hr_" not in f]
        outputs = {}
        for file_name in files:
            lr_test = np.load(os.path.join(input_dir, file_name)).astype("float32")
            if lr_test.shape[1] == 1:
                lr_test = np.moveaxis(lr_test, 1, -1)
            # lr_test = np.minimum(lr_test, hr_train_cutoff)
            #Scale values
            # max_hr_test = np.max(lr_test)
            # lr_test = lr_test/hr_train_cutoff
            outputs[file_name.split(".")[0]] = lr_test
        return outputs
    
    

def train_the_model(args) -> None:
    if args.restart_training:
        clear_dir(args.save_dir)

    NUM_EPOCHS = args.epochs
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    N_REPLICAS = strategy.num_replicas_in_sync
    pprint(f"Num gpus to be used: {N_REPLICAS}")
    BATCH_SIZE = args.batch_size_per_gpu * N_REPLICAS

    create_directories(args.save_dir)
    
    lr_train, hr_train, lr_valid, hr_valid = load_training_data(args.input, args.ratio)

    with open(f"{args.save_dir}/commandline_args.json", 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    train_data_shape = lr_train.shape
    steps_per_epoch = len(lr_train) // BATCH_SIZE
    validation_steps = len(lr_valid) // BATCH_SIZE

    
    train_dataset = get_dataset(lr_train, BATCH_SIZE,
                                strategy=strategy,
                                hr_set=hr_train,
                                training=True)
    valid_dataset = get_dataset(lr_valid, BATCH_SIZE,
                                strategy=strategy,
                                hr_set=hr_valid,
                                training=False)
    del lr_valid, hr_valid, lr_train, hr_train
    K.clear_session()
    callbacks = create_callbacks(save_path=f"{args.save_dir}/models/checkpoints/cp.ckpt")
    model_args = {
        "beta1": args.beta1,
        "lr": args.lr
    }
    with strategy.scope():
        model = create_model(model_args)
        # if args.model_summary:
        #     model.m_module.summary()
        history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch,
                            epochs=NUM_EPOCHS,
                            validation_data=valid_dataset,
                            validation_steps=validation_steps,
                            callbacks=callbacks, verbose=2)
        model.save(f"{args.save_dir}/models/final_model.ckpt")
    pass


def impute_the_target(args):
    if os.path.exists(f"{args.save_dir}/commandline_args.json"):
        with open(f"{args.save_dir}/commandline_args.json", 'r') as f:
            training_args = json.load(f)
        # Ensure that the same cut-off of the training set is used for dealing with outlier reads
        # args.train_cutoff = training_args["train_cutoff"]

    BATCH_SIZE = args.batch_size_per_gpu  # * N_REPLICAS
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    N_REPLICAS = strategy.num_replicas_in_sync
    pprint(f"Num gpus to be used: {N_REPLICAS}")
    BATCH_SIZE = args.batch_size_per_gpu * N_REPLICAS

    K.clear_session()
    model = tf.keras.models.load_model(
        f"{args.save_dir}/models/final_model.ckpt",
        custom_objects=custom_objects,
        compile=False
    )

    lr_test_dict = load_test_data(args.input, args.ratio)
    for keey, vall in lr_test_dict.items():
        print(f"Predicting {keey} with shape of {vall.shape}")

        AUTO = tf.data.AUTOTUNE
        test_dataset = tf.data.Dataset.from_tensor_slices((vall))\
                    .prefetch(AUTO)\
                    .batch(BATCH_SIZE, drop_remainder=False, num_parallel_calls=AUTO)
        steps = int(np.ceil(vall.shape[0]/BATCH_SIZE))
        test_preds = np.round(model.m_module.predict(test_dataset, steps=steps)).astype(np.float32)
        # for ii in range(len(test_preds)):
        #     print(f"Output[{ii}] shape: {test_preds[ii].shape}")
        # with gzip.GzipFile(f"{args.save_dir}/out/test_preds.npy.gz", "w") as f:
        np.save(f"{args.save_dir}/out/preds_{keey}.npy", arr=test_preds)
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
    deciding_args_parser = argparse.ArgumentParser(description='ShiLab\'s TruHiC.', add_help=False)

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
    parser.add_argument('--model-summary', type=str, required=False,
                                      help='Prints the model summary.',
                                      choices=['false', 'true', '0', '1'], default='0')
    parser.add_argument('--epochs', type=int, required=False, help='Maximum number of epochs (default 100)',
                        default=500)
    parser.add_argument('--lr', type=float, required=False, help='Learning Rate (default 0.001)', default=0.001)
    parser.add_argument('--beta1', type=float, required=False, help='Beta1 (default 0.9)', default=0.9)
    parser.add_argument('--batch-size-per-gpu', type=int, required=False, help='Batch size per gpu(default 16)',
                        default=128)

    args = parser.parse_args()
    args.restart_training = str_to_bool(args.restart_training)
    args.model_summary = str_to_bool(args.model_summary)

    if not (args.save_dir.startswith("./") or args.save_dir.startswith("/")):
        args.save_dir = f"./{args.save_dir}"
    pprint(f"Save directory will be:\t{args.save_dir}")

    if args.mode == 'train':
        train_the_model(args)
    else:
        impute_the_target(args)


if __name__ == '__main__':
    main()
