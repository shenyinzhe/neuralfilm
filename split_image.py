import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image


def image2cols(image:np.array, patch_size: list, stride:int):
    """
    https://zhuanlan.zhihu.com/p/39361808
    """
    imhigh, imwidth, imch = image.shape

    # patch indexes
    range_y = np.arange(0, imhigh - patch_size[0], stride)
    range_x = np.arange(0, imwidth - patch_size[1], stride)
    if range_y[-1] != imhigh - patch_size[0]:
        range_y = np.append(range_y, imhigh - patch_size[0])
    if range_x[-1] != imwidth - patch_size[1]:
        range_x = np.append(range_x, imwidth - patch_size[1])
    sz = len(range_y) * len(range_x)  # number of patches

    res = np.zeros((sz, patch_size[0], patch_size[1], imch))

    index = 0
    for y in range_y:
        for x in range_x:
            patch = image[y:y + patch_size[0], x:x + patch_size[1]]
            res[index] = patch
            index = index + 1

    return res.astype(np.uint8)

def main(data_dir: str, size: int):
    input_dir = os.path.join(data_dir, "input")
    label_dir = os.path.join(data_dir, "label")
    images = sorted(os.listdir(input_dir))
    target = sorted(os.listdir(label_dir))

    name = os.path.basename(data_dir)
    new_name = name + "_split"

    new_data_dir = os.path.join(os.path.dirname(data_dir), new_name)

    if not os.path.exists(new_data_dir):
        os.mkdir(new_data_dir)
        os.mkdir(os.path.join(new_data_dir,"input"))
        os.mkdir(os.path.join(new_data_dir, "label"))
    else:
        raise RuntimeError("Dataset exists")

    counter = 1
    for i in tqdm(images):
        im = np.array(Image.open(os.path.join(input_dir, i)))
        im = image2cols(im, [size,size], size)
        for j in im:
            patch = Image.fromarray(j)
            patch.save(os.path.join(new_data_dir, "input",f"{counter}.jpg"))
            counter += 1

    counter = 1
    for i in tqdm(target):
        im = np.array(Image.open(os.path.join(label_dir, i)))
        im = image2cols(im, [size,size], size)
        for j in im:
            patch = Image.fromarray(j)
            patch.save(os.path.join(new_data_dir, "label",f"{counter}.jpg"))
            counter += 1

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Split images into patches')
    args.add_argument('-d', '--data_dir', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--size', default=112, type=int,
                      help='patch size (default: 112)')
    args = args.parse_args()
    main(args.data_dir, args.size)
