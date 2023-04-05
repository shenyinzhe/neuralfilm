import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from empatches import EMPatches


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
        os.mkdir(os.path.join(new_data_dir, "input"))
        os.mkdir(os.path.join(new_data_dir, "label"))
    else:
        raise RuntimeError("Dataset exists")

    emp = EMPatches()
    counter = 1
    for i in tqdm(images):
        im = np.array(Image.open(os.path.join(input_dir, i)))
        im, indices = emp.extract_patches(im, patchsize=size, overlap=0)
        im = np.stack(im, axis=0)
        for j in im:
            patch = Image.fromarray(j)
            patch.save(os.path.join(new_data_dir, "input", f"{counter}.jpg"))
            counter += 1

    counter = 1
    for i in tqdm(target):
        im = np.array(Image.open(os.path.join(label_dir, i)))
        im, indices = emp.extract_patches(im, patchsize=size, overlap=0)
        im = np.stack(im, axis=0)
        for j in im:
            patch = Image.fromarray(j)
            patch.save(os.path.join(new_data_dir, "label", f"{counter}.jpg"))
            counter += 1


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Split images into patches")
    args.add_argument(
        "-d",
        "--data_dir",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-s", "--size", default=112, type=int, help="patch size (default: 112)"
    )
    args = args.parse_args()
    main(args.data_dir, args.size)
