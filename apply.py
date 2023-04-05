import argparse
import os
import shutil
import torch
import numpy as np
from tqdm import tqdm
from empatches import EMPatches
from PIL import Image

import data_loader.data_loaders as module_data
import model.model as module_arch
import model.loss as module_loss
import model.metric as module_metric


from parse_config import ConfigParser
from tempfile import TemporaryDirectory
from torchvision.utils import save_image

# Fix path issue on Windows
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# Fix path issue on Windows


def main(img_path, config):
    with TemporaryDirectory() as dirname:
        # make dirs needed in following operations
        os.mkdir(os.path.join(dirname, "input"))
        os.mkdir(os.path.join(dirname, "output"))

        im = np.array(Image.open(img_path))

        emp = EMPatches()
        im, indices = emp.extract_patches(im, patchsize=224, overlap=0.1)
        im = np.stack(im, axis=0)
        counter = 0
        for patch in im:
            patch = Image.fromarray(patch)
            patch.save(os.path.join(dirname, "input", f"{counter}.jpg"))
            counter += 1
        # repeat input data in the label folder, just to reuse the standard dataloader
        shutil.copytree(os.path.join(dirname, "input"), os.path.join(dirname, "label"))

        data_loader = getattr(module_data, config["data_loader"]["type"])(
            dirname,
            batch_size=64,
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=2,
        )

        # build model architecture
        model = config.init_obj("arch", module_arch)

        checkpoint = torch.load(config.resume)
        state_dict = checkpoint["state_dict"]
        if config["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        # prepare model for testing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)

                # save the output patches
                counter = len(os.listdir(os.path.join(dirname, "output")))
                for j in range(len(output)):
                    save_image(
                        output[j], os.path.join(dirname, "output", f"{counter}.jpg")
                    )
                    counter += 1

        # read the patches as a numpy array
        patches = sorted(
            os.listdir(os.path.join(dirname, "output")),
            key=lambda x: int(x.split(".")[0]),
        )
        patches_np = []
        for patch in patches:
            patch = np.array(Image.open(os.path.join(dirname, "output", patch)))
            patches_np.append(patch)
        patches_np = np.stack(patches_np)

    # recover the image from patches
    img = emp.merge_patches(patches_np, indices, mode="avg").astype(np.uint8)
    img = Image.fromarray(img)
    dirname = os.path.dirname(img_path)
    filename = os.path.basename(img_path).split(".")[0]
    img.save(os.path.join(dirname, filename + "_applied.jpg"))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Split images into patches")
    args.add_argument(
        "-i", "--image", default=None, type=str, help="config file path (default: None)"
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to model checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    # TODO remove the config arg
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    config = ConfigParser.from_args(args)
    args = args.parse_args()
    main(args.image, config)
