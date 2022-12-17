import os
import matplotlib.image as mpimg
import numpy as np
from utils import manage_dir
from PIL import Image
import math
import cv2


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
                       dargs['overlapping'], realtime_update=False)
    else:
        print(f"Data ALREADY exists at {DATA_FOLDER_DIR}")


def save_one_chunk(DATA_FOLDER_DIR, METADATA_FOLDER_DIR, MASKDATA_FOLDER_DIR,
                   n_classes, n_samples, data_mode='train', n_instances=1, random_n_instances=False,
                   type_noise=False, overlapping=False, realtime_update=False):
    if n_classes==10:
        from objgen.random_simple_gen_implemented import TenClassesPyIO
        dataset = TenClassesPyIO(n_instances=n_instances,
                                 random_n_instances=random_n_instances,
                                 type_noise=type_noise, overlapping=overlapping)
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
        bbox_list = []
        image_id = f'SYNTHETIC_{data_mode}_{i + 1}.png'
        # merge image instance layers
        ep = 1e-2
        images = x
        bg = images[-1]
        cimg = np.zeros(bg.shape)
        n_generated_instances = images.shape[0]
        if overlapping:
            # leave out background
            _images = images[:-1]
            # merge image layers
            for img in _images[::-1]:
                mask = img > ep
                cimg[mask] = img[mask]

                # merge segment layers
                heatmap = np.zeros(h[0].shape)
                for _h in h[::-1]:
                    mask = _h > ep
                    heatmap[mask] = _h[mask]

            # metadata: bounding boxes
            # take into account that multiple instances are layerd
            # e.g. instance 1 on top of instance 2 -> bounding box instance may be clipped by instance 1
            # e.g. instance 1 on top of instance 2, 2 on 3, 3 on 4 -> bbox 4 clipped by instances 1,2,3
            ignore_mask = np.ones(h[0].shape) > 0  # used to mask upper-layer overlapping segments
            for _h in h.copy():
                _h *= ignore_mask.astype('uint8')
                _x, _y = np.ma.where(_h > 0)
                if len(_x) > 0:
                    xmin, xmax = min(_x), max(_x)
                    ymin, ymax = min(_y), max(_y)
                    # store location
                    locations.append(f'{data_mode}/{image_id},{xmin},{ymin},{xmax},{ymax}')
                    bbox_list.append((xmin, ymin, xmax, ymax))
                # segment ignore mask of current layer
                _h_ignore_mask = np.logical_not(_h > 0)
                # merge with upper layers ignore mask
                ignore_mask = np.logical_and(ignore_mask, _h_ignore_mask)

        else:
            # Generate a square number of tiles that can contain all instances.
            # The size of 1 tile equals the final image size.
            tile_dim = math.ceil(math.sqrt(n_generated_instances))
            n_tiles =  tile_dim ** 2
            # randomly select tiles for object instances
            image_tile_idx = np.random.permutation(n_tiles)[:n_generated_instances]
            # allocate n_tiles number of image locations
            tiles = np.zeros((n_tiles,) + bg.shape)
            # leave out background
            _images = images[:-1]
            # distribute images over selected tiles
            for idx, img in enumerate(_images):
                tiles[image_tile_idx[idx]] = img
            # reshape in tile_dim x tile_dim x H x W x 3
            tile_shape = (tile_dim, tile_dim) + bg.shape
            img = np.concatenate(np.concatenate(tiles.reshape(tile_shape), axis=1), axis=1)
            # resize to image shape
            resize = bg.shape[:2]
            cimg = cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)

            # merge semgent tiles
            # allocate n_tiles number of segment locations
            h_tiles = np.zeros((n_tiles,) + h[0].shape)
            # distribute semgents over selected tiles
            for idx, _h in enumerate(h):
                h_tiles[image_tile_idx[idx]] = _h
            # reshape in tile_dim x tile_dim x H x W
            h_tile_shape = (tile_dim, tile_dim) + h[0].shape
            h_img = np.concatenate(np.concatenate(h_tiles.reshape(h_tile_shape), axis=1), axis=1)
            # resize to image shape
            resize = h[0].shape
            heatmap = cv2.resize(h_img, resize, interpolation=cv2.INTER_LINEAR)

            # metadata: bounding boxes
            for idx, _h in enumerate(h):
                # get bounding box of segments before resizing
                _x, _y = np.ma.where(_h > 0)
                xmin, xmax = min(_x), max(_x)
                ymin, ymax = min(_y), max(_y)
                # transform bounding box
                # add tile offsets
                tile_x = image_tile_idx[idx] // tile_dim
                tile_y = image_tile_idx[idx] % tile_dim
                xmin += tile_x * _h.shape[0]
                xmax += tile_x * _h.shape[0]
                ymin += tile_y * _h.shape[1]
                ymax += tile_y * _h.shape[1]
                # resize boxes
                xmin //= tile_dim
                xmax //= tile_dim
                ymin //= tile_dim
                ymax //= tile_dim
                # store location
                locations.append(f'{data_mode}/{image_id},{xmin},{ymin},{xmax},{ymax}')
                bbox_list.append((xmin, ymin, xmax, ymax))

        # merge background into image
        pos = np.stack(((cimg[:,:,0]<ep),(cimg[:,:,1]<ep),(cimg[:,:,2]<ep))).transpose((1,2,0))
        cimg =  cimg + pos * bg
        cimg = np.clip(cimg, a_min=0., a_max=1.)

        # metadata: store image path
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

        # bboxdata: save image with bounding box
        bbox_id = f'SYNTHETIC_{data_mode}_{i + 1}_bbox.png'
        bbox_path = os.path.join(MASKDATA_FOLDER_DIR, bbox_id)
        bbox_img = cimg.copy()
        d = 3
        for bbox in bbox_list:
            xmin, ymin, xmax, ymax = bbox
            bbox_mask = np.zeros(heatmap.shape) > 0
            bbox_mask[xmin:xmax + 1, ymin:ymax + 1] = True
            bbox_mask[xmin+d:xmax-d, ymin+d:ymax-d] = False
            bbox_img[bbox_mask] = 1.0
        mpimg.imsave(bbox_path, bbox_img)

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

   # metadata: write bounding box locations
    locations_path = os.path.join(METADATA_FOLDER_DIR, 'localization.txt')
    # TODO support for missing ignore mask in dataloaders.py
    with open(locations_path, 'w') as f:
        f.writelines('\n'.join(locations))