import os
import argparse
import lmdb
import pickle
import tqdm
from PIL import Image

def get_image_ids(metadata):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    image_ids = []
    with open(metadata) as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))
    return image_ids


def meta2lmdb(metadata_path, data_root, lmdb_path):
    print(f"Loading metadata from {metadata_path}")
    image_ids = get_image_ids(metadata_path)
    isdir = os.path.isdir(lmdb_path)
    print(f"Generate LMDB to {lmdb_path}")
    db = lmdb.open(lmdb_path, subdir=isdir,
                   # map size = 256 GB
                   map_size=256 * 1024**3, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    for index in tqdm.tqdm(range(len(image_ids)), total=len(image_ids), desc='images'):
        image_id = image_ids[index]
        path = os.path.join(data_root, image_id)
        with open(path, 'rb') as f:
            raw = f.read()
        # raw = image.tobytes()
        key = u'{}'.format(index).encode('ascii')
        value = pickle.dumps(raw)
        txn.put(key, value)
    txn.commit()

    # finish iterating through dataset
    with db.begin(write=True) as txn:
        txn.put(b'__len__', pickle.dumps(len(image_ids)))
        print(len(image_ids))

    print("Flushing database ...")
    db.sync()
    db.close()


metadata_default = 'data/metadata/ILSVRC/train/image_ids.txt'
data_root = 'data/dataset/ILSVRC'
lmdb_path_default = 'lmdb_train.lmdb'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, default=metadata_default, help='meta data path')
    parser.add_argument('--data_root', type=str, default=data_root, help='image data root path')
    parser.add_argument('--lmdb_path', type=str, default=lmdb_path_default, help='lmdb path')
    args = parser.parse_args()
    meta2lmdb(args.metadata, args.data_root, args.lmdb_path)