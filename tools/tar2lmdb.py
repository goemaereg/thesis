import tarfile
import os
import argparse
import lmdb
import pickle
import tqdm


def tar2lmdb(tar_path, name, key_prefix):
    print(f"Loading dataset from {tar_path}")

    lmdb_path = f"{name}.lmdb"
    isdir = os.path.isdir(lmdb_path)

    print(f"Generate LMDB to {lmdb_path}")
    db = lmdb.open(lmdb_path, subdir=isdir,
                   # map size = 256 GB
                   map_size=256 * 1024**3, readonly=False,
                   meminit=False, map_async=True)

    tar = tarfile.TarFile(tar_path, 'r')
    tar_names = sorted(tar.getnames())
    count = 0
    for tar_fname in tqdm.tqdm(tar_names, total=len(tar_names), desc='tar dirs'):
        tar2 = tarfile.TarFile(fileobj=tar.extractfile(tar_fname))
        tar2_names = sorted(tar2.getnames())
        txn = db.begin(write=True)
        for image_fname in tar2_names:
            image = tar2.extractfile(image_fname).read()
            image_class = tar_fname.split('.')[0]
            key = os.path.join(image_class, image_fname)
            if key_prefix:
                key = os.path.join(key_prefix, key)
            key = u'{}'.format(key).encode('ascii')
            value = pickle.dumps(image)
            txn.put(key, value)
            count += 1
        txn.commit()

    # finish iterating through dataset
    with db.begin(write=True) as txn:
        txn.put(b'__len__', pickle.dumps(count))
        print(count)

    print("Flushing database ...")
    db.sync()
    db.close()


tar_path_default = 'data/dataset/ILSVRC/ILSVRC2012_img_train.tar'
lmdb_path_default = 'lmdb_train'
key_prefix_default = 'train'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tar_path', type=str, default=tar_path_default, help='tar path')
    parser.add_argument('--lmdb_path', type=str, default=lmdb_path_default, help='lmdb path')
    parser.add_argument('--key_prefix', type=str, default=key_prefix_default, help='lmdb key prefix')
    args = parser.parse_args()
    tar2lmdb(args.tar_path, args.lmdb_path, args.key_prefix)