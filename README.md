# Neural Film
Emulate any color style if you have the training data :)


## Usage
- install required packages
- Prepare your dataset
  - An example (recommended): export original images from Lightroom as inputs, apply a specific camera profile and then export as labels
  - IMPORTANT: the corresponding input and label must have the same filename
- Place the dataset according to the following structure
  ```
  neuralfilm/
  │
  ├─ data/
  │    ├─ <your_dataset>/ - put your training data here
  │    │    ├─ input/
  │    │    └─ label/
  │    └─ <your_dataset_split>/ - will appear once you run split_image.py
  └── ...
  ```
- Split images into patches 
  ```
  python split_image.py -d data/<your_dataset>
  ```
- Modify `data_dir` in `config.json`: ```"data_dir": "data/<your_dataset_split>",```
- Train `python train.py -c config.json`
- Apply the learned filter to any images 
  ```
  python apply.py -i <input_image>.jpg -r saved/models/neuralfilm/<time>/model_best.pth
  ```
  - Recommend that `<input_image>.jpg` is obtained in the same way as the training input (such as in the above example: export from Lightroom without editing)


## Acknowledgements
Created from [PyTorch Template Project](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque)
