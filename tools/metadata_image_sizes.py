import os
import argparse
import cv2
import tqdm


def main(dir_images, fname_sizes):
    images = sorted(os.listdir(dir_images))
    prefix = 'val'
    lines = []
    for index, image in enumerate(tqdm.tqdm(images, total=len(images), desc='parse images')):
        image_id = os.path.join(prefix, image)
        img = cv2.imread(os.path.join(dir_images, image))
        h, w, _ = img.shape
        line = f'{image_id},{w},{h}'
        if index < len(images) - 1:
            line += '\n'
        lines.append(line)
    with open(fname_sizes, 'w') as fp:
        fp.writelines(lines)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, help='directory path to images')
    parser.add_argument('--sizes', type=str, help='file path to store image sizes')
    parser.add_argument('--prefix', type=str, default='val', help='prefix for the image id')
    args = parser.parse_args()
    main(args.images, args.sizes)