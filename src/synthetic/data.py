import os
import matplotlib.image as mpimg
import numpy as np
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
                       dargs['n_instances'], dargs['random_n_instances'], dargs['type_noise'],
                       realtime_update=False)
    else:
        print(f"Data ALREADY exists at {DATA_FOLDER_DIR}")


def save_one_chunk(DATA_FOLDER_DIR, METADATA_FOLDER_DIR, MASKDATA_FOLDER_DIR,
                   n_classes, n_samples, data_mode='train', n_instances=1, random_n_instances=False,
                   type_noise=False, realtime_update=False):
    if n_classes==10:
        from objgen.random_simple_gen_implemented import TenClassesPyIO
        dataset = TenClassesPyIO(n_instances=n_instances,
                                 random_n_instances=random_n_instances,
                                 type_noise=type_noise)
    elif n_classes==3:
        from objgen.random_simple_gen_implemented2 import ThreeClassesPyIO
        dataset = ThreeClassesPyIO()
    else:
        raise NotImplementedError()

    dataset.setup_xai_evaluation_0001(general_meta_setting=None, explanation_setting=None,
        data_size=n_samples, realtime_update=realtime_update)
    image_ids = []
    class_labels = []
    segment_masks = []
    locations = []
    for i in range(dataset.__len__()):
        x, y0 = dataset.__getitem__(i)
        h = dataset.h[i]  # heatmap

        # merge image instance layers
        ep = 1e-2
        images = x
        bg = images[-1]
        cimg = np.zeros(bg.shape)
        images = images[:-1] # leave out background
        for img in images[::-1]:
            mask = img > ep
            cimg[mask] = img[mask]

        # merge background into image
        pos = np.stack(((cimg[:,:,0]<ep),(cimg[:,:,1]<ep),(cimg[:,:,2]<ep))).transpose((1,2,0))
        cimg =  cimg + pos * bg
        cimg = np.clip(cimg, a_min=0., a_max=1.)

        # merge heatmap layers
        heatmap = np.zeros(h[0].shape)
        for _h in h[::-1]:
            mask = _h > ep
            heatmap[mask] = _h[mask]

        # metadata: store image path
        image_id = f'SYNTHETIC_{data_mode}_{i + 1}.png'
        image_ids.append(f'{data_mode}/{image_id}')

        # metadata: store image class label
        class_labels.append(f'{data_mode}/{image_id},{str(y0)}')
        image_path = os.path.join(DATA_FOLDER_DIR, image_id)

        # dataset: save RGB image
        # matplotlib.image.imsave requires RGB image shape as (H,W,C)
        mpimg.imsave(image_path, cimg)

        # metadata: store segmentation mask path
        mask_id = image_id
        segment_masks.append(f'{data_mode}/{image_id},{data_mode}/{mask_id}')

        # maskdata: save B/W segmentation mask
        heatmap = (heatmap > 0.0).astype('uint8') * 255
        mask_path = os.path.join(MASKDATA_FOLDER_DIR, mask_id)
        img = Image.fromarray(heatmap)
        img.save(mask_path)

        # metadata: bounding boxes
        # take into account that multiple instances are layerd
        # e.g. instance 1 on top of instance 2 -> bounding box instance may be clipped by instance 1
        # e.g. instance 1 on top of instance 2, 2 on 3, 3 on 4 -> bbox 4 clipped by bbox 1,2,3
        shape = heatmap.shape
        ignore_mask = np.ma.ones(shape) # used to wipe out hidden parts of lower-layer heatmaps
        for _h in h:
            _h *= ignore_mask.astype('uint8')
            _x, _y = np.ma.where(_h > 0)
            xmin, xmax = min(_x), max(_x)
            ymin, ymax = min(_y), max(_y)
            # bbox_mask
            bbox_neg_mask = np.ma.ones(shape)
            bbox_neg_mask[xmin:xmax+1,ymin:ymax+1] = False
            # negation mask
            ignore_mask = np.logical_and(ignore_mask, bbox_neg_mask)
            # store location
            locations.append(f'{data_mode}/{image_id},{xmin},{ymin},{xmax},{ymax}')

    # metadata: write image paths
    image_ids_path = os.path.join(METADATA_FOLDER_DIR, 'image_ids.txt')
    with open(image_ids_path, 'w') as f:
        f.writelines('\n'.join(image_ids))

    # metadata: write class labels
    class_labels_path = os.path.join(METADATA_FOLDER_DIR, 'class_labels.txt')
    with open(class_labels_path, 'w') as f:
        f.writelines('\n'.join(class_labels))

    # metadata: write segmentation mask paths
    segment_masks_path = os.path.join(METADATA_FOLDER_DIR, 'segment_masks.txt')
    # TODO support for missing ignore mask in dataloaders.py
    with open(segment_masks_path, 'w') as f:
        f.writelines('\n'.join(segment_masks))

   # metadata: write locations
    locations_path = os.path.join(METADATA_FOLDER_DIR, 'localization.txt')
    # TODO support for missing ignore mask in dataloaders.py
    with open(locations_path, 'w') as f:
        f.writelines('\n'.join(locations))