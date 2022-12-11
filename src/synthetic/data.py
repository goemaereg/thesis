import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import joblib
from skimage.transform import resize
from utils import manage_dir
from PIL import Image


def prepare_data(dargs):
    print('prepare_data...')

    DIRS = manage_dir(dargs)
    DATA_MODES = {
        'train': 'DATA_TRAIN_FOLDER_DIR',
        'val': 'DATA_VAL_FOLDER_DIR',
        'test': 'DATA_TEST_FOLDER_DIR',
    }
    METADATA_MODES = {
        'train': 'METADATA_TRAIN_FOLDER_DIR',
        'val': 'METADATA_VAL_FOLDER_DIR',
        'test': 'METADATA_TEST_FOLDER_DIR',
    }
    MASKDATA_MODES = {
        'train': 'MASKDATA_TRAIN_FOLDER_DIR',
        'val': 'MASKDATA_VAL_FOLDER_DIR',
        'test': 'MASKDATA_TEST_FOLDER_DIR',
    }
    DATA_FOLDER_DIR = DIRS[DATA_MODES[dargs['data_mode']]]
    METADATA_FOLDER_DIR = DIRS[METADATA_MODES[dargs['data_mode']]]
    MASKDATA_FOLDER_DIR = DIRS[MASKDATA_MODES[dargs['data_mode']]]

    if len(os.listdir(DATA_FOLDER_DIR))==0:
        print(f'preparing data to save')
        save_one_chunk(DATA_FOLDER_DIR, METADATA_FOLDER_DIR, MASKDATA_FOLDER_DIR,
                       dargs['n_classes'], dargs['n_samples'], dargs['data_mode'],
                       dargs['max_instances'], dargs['type_noise'],
                       realtime_update=False)
    else:
        print(f"Data ALREADY exists at {DATA_FOLDER_DIR}")
        # SHARD_NAME = f'data-{str(dargs["n_classes"])}'
        # DATA_DIR = os.path.join(DATA_FOLDER_DIR, str(SHARD_NAME))
        # display_some_shard_samples(DATA_DIR, DATA_FOLDER_DIR, name=f'samples')


def save_one_chunk(DATA_FOLDER_DIR, METADATA_FOLDER_DIR, MASKDATA_FOLDER_DIR,
                   n_classes, n_samples, data_mode='train', max_instances=1,
                   type_noise=False, realtime_update=False):
    if n_classes==10:
        from objgen.random_simple_gen_implemented import TenClassesPyIO
        dataset = TenClassesPyIO(max_instances=max_instances, type_noise=type_noise)
    elif n_classes==3:
        from objgen.random_simple_gen_implemented2 import ThreeClassesPyIO
        dataset = ThreeClassesPyIO()
    else:
        raise NotImplementedError()

    dataset.setup_xai_evaluation_0001(general_meta_setting=None, explanation_setting=None,
        data_size=n_samples, realtime_update=realtime_update)
    image_ids = []
    class_labels = []
    localizations = []
    for i in range(dataset.__len__()):
        x, y0 = dataset.__getitem__(i)
        h = dataset.h[i]  # heatmap
        # save image
        image_id = f'SYNTHETIC_{data_mode}_{i + 1}.png'
        image_ids.append(f'{data_mode}/{image_id}')
        class_labels.append(f'{data_mode}/{image_id},{str(y0)}')
        image_path = os.path.join(DATA_FOLDER_DIR, image_id)
        mpimg.imsave(image_path, x.transpose((1,2,0)))
        # TODO multiple segmentation masks support
        # save segmentation mask
        mask_id = image_id
        localization = f'{data_mode}/{image_id},{data_mode}/{mask_id}'
        localizations.append(localization)
        mask_path = os.path.join(MASKDATA_FOLDER_DIR, mask_id)
        # binarize heatmap
        h = (h > 0.0).astype('uint8')*255
        img = Image.fromarray(h)
        img.save(mask_path)
    # write metadata
    image_ids_path = os.path.join(METADATA_FOLDER_DIR, 'image_ids.txt')
    with open(image_ids_path, 'w') as f:
        f.writelines('\n'.join(image_ids))
    class_labels_path = os.path.join(METADATA_FOLDER_DIR, 'class_labels.txt')
    with open(class_labels_path, 'w') as f:
        f.writelines('\n'.join(class_labels))
    localization_path = os.path.join(METADATA_FOLDER_DIR, 'localization.txt')
    # TODO support for missing ignore mask in dataloaders.py
    with open(localization_path, 'w') as f:
        f.writelines('\n'.join(localizations))

def display_some_shard_samples(SHARD_DIR, SHARD_FOLDER_DIR, name):
    SAMPLES_DIR = os.path.join(SHARD_FOLDER_DIR, 'samples_display')
    if not os.path.exists(SAMPLES_DIR):
        os.makedirs(SAMPLES_DIR, exist_ok=True)

    print('\nsource:', SHARD_DIR)
    dataset = load_dataset_from_a_shard(SHARD_DIR, reshape_size=None)
    nshow = np.min([4, dataset.__len__()])

    plt.figure(figsize=(8, 4))
    for i in range(nshow):
        x, y0 = dataset.__getitem__(i)
        h = dataset.h[i]  # heatmap
        print(f'x.shape: {x.shape} | y0:{y0} | h.shape: {h.shape} | v["type"]: {dataset.v[i]["type"]} ')
        if i == 0: print('  v:', dataset.v[i].keys())

        plt.gcf().add_subplot(2, nshow, i + 1)
        plt.gca().imshow(x.transpose(1, 2, 0), vmin=0, vmax=1)
        plt.gca().set_title(f'y0:{y0}')
        plt.gcf().add_subplot(2, nshow, i + 1 + nshow)
        plt.gca().imshow(h, vmin=-1., vmax=1, cmap='bwr')
    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLES_DIR, name + '.png'))


def load_dataset_from_a_shard(SHARD_DIR, reshape_size=None):
    # reshape_size : (C, H, W) tuple
    this_dataset = joblib.load(SHARD_DIR)
    this_dataset.x = np.array(this_dataset.x)  # original shape (N, C, H, W)
    if reshape_size is not None:
        s, N = reshape_size, len(this_dataset.x)
        temp_x = []

        size_HW = reshape_size[1:]
        temp_h = []

        for i in range(N):
            temp = this_dataset.x[i].transpose(1, 2, 0)
            temp = resize(temp, (s[1], s[2], s[0]))
            temp = temp.transpose(2, 0, 1)
            temp_x.append(temp)

            h_resize = resize(this_dataset.h[i], size_HW)
            temp_h.append(h_resize)

        this_dataset.x = np.array(temp_x)
        this_dataset.h = np.array(temp_h)

    return this_dataset