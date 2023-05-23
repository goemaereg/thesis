import os
import matplotlib.image as mpimg
import numpy as np
from utils import manage_dir
from PIL import Image
import math
import cv2
import tqdm
from torch.utils.data import Dataset

def create_context(args):
    tag1 = 'o' if args['overlapping'] else 'd'
    tag2 = '0' if args['random_n_instances'] else str(args['n_instances'])
    tag3 = 'b' if args['background'] else 't'
    tags = [tag1, tag2, tag3]
    DIRS = manage_dir(args, tags)
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
    context = {
        'tags': tags
    }
    split_values = [('train', 1000), ('val', 200), ('test', 200)]
    for split, num_images in split_values:
        context[split] = {
            'num_images': num_images,
            'data': DIRS[DATA_MODES[split]],
            'metadata': DIRS[METADATA_MODES[split]],
            'maskdata': DIRS[MASKDATA_MODES[split]],
            'image_ids': [],
            'image_sizes': [],
            'class_labels': [],
            'segment_masks': [],
            'locations': []
        }
    return context

def create_image_dataset(n_classes:int, n_images, n_instances:int = 1, random_n_instances:bool = False,
                   type_noise:bool = False, overlapping:bool = False, realtime_update:bool = False) -> Dataset:
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
        data_size=n_images, realtime_update=realtime_update)
    return dataset

def prepare_data(dargs):
    splits = ['train', 'val', 'test']
    dargs['background'] = True
    ctxt_bg = create_context(dargs)
    dargs['background'] = False
    ctxt_tr = create_context(dargs)

    for split in splits:
        if len(os.listdir(ctxt_bg[split]['data'])) != 0:
            print(f"Data ALREADY exists at {ctxt_bg[split]['data']}. Skipping...")
            continue
        if len(os.listdir(ctxt_tr[split]['data'])) != 0:
            print(f"Data ALREADY exists at {ctxt_bg[split]['data']}. Skipping...")
            continue

        print(f'Generating images for {split} dataset.')
        n_images = ctxt_bg[split]['num_images']
        n_classes = 10
        n_instances = dargs['n_instances']
        dataset = create_image_dataset(n_classes=n_classes, n_images=n_images, n_instances=n_instances,
                                       random_n_instances=False, type_noise=False, overlapping=False,
                                       realtime_update=False)
        print(f'Post-processing images for {split} dataset.')
        save_one_chunk(ctxt_bg, ctxt_tr, split, dataset)

def save_one_chunk(ctxt_bg, ctxt_tr, split, dataset, overlapping=False, background=True):
    tq0 = tqdm.tqdm(range(dataset.__len__()), total=dataset.__len__(), desc='Generate meta data')
    for i in tq0:
        x, y0 = dataset.__getitem__(i)
        h = dataset.h[i]  # heatmap
        bbox_list = []
        image_id_bg = f"{'_'.join(['SYNTHETIC'] + ctxt_bg['tags'] + [split, str(i + 1)])}.png"
        image_id_tr = f"{'_'.join(['SYNTHETIC'] + ctxt_tr['tags'] + [split, str(i + 1)])}.png"
        # merge image instance layers
        ep = 1e-2
        images = x
        bg = images[-1] if background else np.zeros(images[-1].shape)
        cimg = np.zeros(bg.shape)
        # cimg is a stacked array of instance images + background image
        # background image doesn't count as instance
        n_generated_instances = images.shape[0] - 1
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
            # take into account that multiple instances are layered
            # e.g. instance 1 on top of instance 2 -> bounding box instance may be clipped by instance 1
            # e.g. instance 1 on top of instance 2, 2 on 3, 3 on 4 -> bbox 4 clipped by instances 1,2,3
            ignore_mask = np.ones(h[0].shape) > 0  # used to mask upper-layer overlapping segments
            for _h in h.copy():
                _h *= ignore_mask.astype('uint8')
                _rows, _cols = np.ma.where(_h > 0)
                if len(_cols) > 0:
                    # locations are stored in (x,y) notation: x:colum, y:row
                    xmin, xmax = min(_cols), max(_cols)
                    ymin, ymax = min(_rows), max(_rows)
                    # store location
                    ctxt_bg[split]['locations'].append(f'{split}/{image_id_bg},{xmin},{ymin},{xmax},{ymax}')
                    ctxt_tr[split]['locations'].append(f'{split}/{image_id_tr},{xmin},{ymin},{xmax},{ymax}')
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
            # randomly select tiles where to place object instances
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

            # merge segment tiles
            # allocate n_tiles number of segment locations
            h_tiles = np.zeros((n_tiles,) + h[0].shape)
            # distribute segments over selected tiles
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
                _rows, _cols = np.ma.where(_h > 0)
                # locations are stored in (x,y) notation: x:colum, y:row
                xmin, xmax = min(_cols), max(_cols)
                ymin, ymax = min(_rows), max(_rows)
                # transform bounding box
                # add tile offsets
                tile_row = image_tile_idx[idx] // tile_dim
                tile_col = image_tile_idx[idx] % tile_dim
                tile_height, tile_width = _h.shape
                tile_x = tile_col * tile_width
                tile_y = tile_row * tile_height
                xmin += tile_x
                xmax += tile_x
                ymin += tile_y
                ymax += tile_y
                # resize boxes
                xmin //= tile_dim
                xmax //= tile_dim
                ymin //= tile_dim
                ymax //= tile_dim
                # store location
                ctxt_bg[split]['locations'].append(f'{split}/{image_id_bg},{xmin},{ymin},{xmax},{ymax}')
                ctxt_tr[split]['locations'].append(f'{split}/{image_id_tr},{xmin},{ymin},{xmax},{ymax}')
                bbox_list.append((xmin, ymin, xmax, ymax))

        # merge background into image
        pos = np.stack(((cimg[:,:,0] < ep), (cimg[:,:,1] < ep), (cimg[:,:,2] < ep))).transpose((1,2,0))
        cimg_tr = cimg.copy()
        cimg = cimg + pos * bg
        cimg = np.clip(cimg, a_min=0., a_max=1.)

        # metadata: store image path
        ctxt_bg[split]['image_ids'].append(f'{split}/{image_id_bg}')
        ctxt_tr[split]['image_ids'].append(f'{split}/{image_id_tr}')

        # metadata: store image size
        ctxt_bg[split]['image_sizes'].append(f'{split}/{image_id_bg},{cimg.shape[0]},{cimg.shape[1]}')
        ctxt_tr[split]['image_sizes'].append(f'{split}/{image_id_tr},{cimg.shape[0]},{cimg.shape[1]}')

        # metadata: store image class label
        ctxt_bg[split]['class_labels'].append(f'{split}/{image_id_bg},{str(y0)}')
        ctxt_tr[split]['class_labels'].append(f'{split}/{image_id_tr},{str(y0)}')

        # dataset: save RGB image
        # matplotlib.image.imsave requires RGB image shape as (H,W,C)
        image_bg_path = os.path.join(ctxt_bg[split]['data'], image_id_bg)
        mpimg.imsave(image_bg_path, cimg)
        image_tr_path = os.path.join(ctxt_tr[split]['data'], image_id_tr)
        mpimg.imsave(image_tr_path, cimg_tr)

        # metadata: store segmentation mask path
        mask_id_bg = image_id_bg
        ctxt_bg[split]['segment_masks'].append(f'{split}/{image_id_bg},{split}/{mask_id_bg},')
        mask_id_tr = image_id_tr
        ctxt_tr[split]['segment_masks'].append(f'{split}/{image_id_tr},{split}/{mask_id_tr},')

        # maskdata: save segmentation mask
        heatmap = (heatmap > 0.0).astype('uint8') * 255
        mask_bg_path = os.path.join(ctxt_bg[split]['maskdata'], mask_id_bg)
        mask_tr_path = os.path.join(ctxt_tr[split]['maskdata'], mask_id_tr)
        img = Image.fromarray(heatmap)
        img.save(mask_bg_path)
        img.save(mask_tr_path)

    # metadata: write image paths
    for ctxt in [ctxt_bg, ctxt_tr]:
        image_ids_path = os.path.join(ctxt[split]['metadata'], 'image_ids.txt')
        with open(image_ids_path, 'w') as f:
            f.writelines('\n'.join(ctxt[split]['image_ids']))

        # metadata: write image sizes
        image_sizes_path = os.path.join(ctxt[split]['metadata'], 'image_sizes.txt')
        with open(image_sizes_path, 'w') as f:
            f.writelines('\n'.join(ctxt[split]['image_sizes']))

        # metadata: write class labels
        class_labels_path = os.path.join(ctxt[split]['metadata'], 'class_labels.txt')
        with open(class_labels_path, 'w') as f:
            f.writelines('\n'.join(ctxt[split]['class_labels']))

        # metadata: write segmentation mask paths
        segment_masks_path = os.path.join(ctxt[split]['metadata'], 'segments.txt')
        with open(segment_masks_path, 'w') as f:
            f.writelines('\n'.join(ctxt[split]['segment_masks']))

       # metadata: write bounding box locations
        locations_path = os.path.join(ctxt[split]['metadata'], 'localization.txt')
        with open(locations_path, 'w') as f:
            f.writelines('\n'.join(ctxt[split]['locations']))