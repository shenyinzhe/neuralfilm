import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from empatches import EMPatches

import data_loader.data_loaders as module_data
import model.model as module_arch
import model.loss as module_loss
import model.metric as module_metric

from PIL import Image
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
        os.mkdir(os.path.join(dirname, "label"))
        os.mkdir(os.path.join(dirname, "output"))

        im = np.array(Image.open(img_path))
        imsize = im.shape[:2]
        # TODO: So far the model is trained with 112x112 patches. Config it in the future
        patch_length = 224
        emp = EMPatches()
        im, indices = emp.extract_patches(im, patchsize=patch_length, overlap=0.1)
        im = np.stack(im,axis=0)
        counter = 0
        for patch in im:
            patch = Image.fromarray(patch)
            patch.save(os.path.join(dirname, "input",f"{counter}.jpg"))
            counter += 1
        # TODO: refactor. Here I repeat input data in the label folder, just to reuse the standard dataloader
        for patch in im:
            patch = Image.fromarray(patch)
            patch.save(os.path.join(dirname, "label",f"{counter}.jpg"))
            counter += 1

        data_loader = getattr(module_data, config['data_loader']['type'])(
            dirname,
            augment=False,
            batch_size=64,
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=2
        )

        # build model architecture
        model = config.init_obj('arch', module_arch)

        # get function handles of loss and metrics
        loss_fn = getattr(module_loss, config['loss'])
        metric_fns = [getattr(module_metric, met) for met in config['metrics']]

        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        total_loss = 0.0
        total_metrics = torch.zeros(len(metric_fns))

        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)

                # save the output patches
                counter = len(os.listdir(os.path.join(dirname, "output")))
                for j in range(len(output)):
                    save_image(output[j], os.path.join(dirname, "output",f"{counter}.jpg"))
                    counter += 1

                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(data_loader.sampler)
        print("average loss:", total_loss / n_samples)

        # read the patches as a numpy array
        patches = sorted(os.listdir(os.path.join(dirname, "output")), key=lambda x:int(x.split(".")[0]))
        patches_np = []
        for patch in patches:
            patch = np.array(Image.open(os.path.join(dirname, "output",patch)))
            patches_np.append(patch)
        patches_np = np.stack(patches_np)

    # recover the image from patches
    img = emp.merge_patches(patches_np, indices, mode='avg').astype(np.uint8)
    img = Image.fromarray(img)
    dirname = os.path.dirname(img_path)
    filename = os.path.basename(img_path).split(".")[0]
    img.save(os.path.join(dirname, filename + "_applied.jpg"))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Split images into patches')
    args.add_argument('-i', '--image', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to model checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    # TODO remove the config arg
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    config = ConfigParser.from_args(args)
    args = args.parse_args()
    main(args.image, config)
