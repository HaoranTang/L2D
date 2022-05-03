# ESE 650 Final Project: Learning to Drive in a Day
##### Qirui Wu, wuqirui@seas; Haoran Tang, thr99@seas
---
#### 0. Requirements
1) Please use Python 3.7 with latest PyTorch and Stable-Baselines 3 for this project, and install Carla 0.9.13 + UE4.
2) Dataset is recorded in Carla and is placed under the root. After starting Carla simulator, simply run manual_control.py to drive manually (also various commands including autopilot), changing camera to first-person and recording screen shots.
#### 1. Train
Please run train.py for training; change arguments in argument parser including number of steps (episodes) and path to last model if necessary. Change hyper-parameters in .yml files from ./hyperparams. Pre-trained VAE model is saved in ./logs as train_epoch_last.pth, but we will not upload ./logs regarding its size. If you need it please let us know so we can share a google drive. If you need to train a new VAE, please run .vae/train.py.
#### 2. Eval
Evaluations will be performed during training, but we also have a test.py which can evaluate the model after training. Simply change the experiment id to load the corresponding model, stored in .logs/THE ALGORITHM/. But we will not upload ./logs regarding its size. If you need it please let us know so we can share.
#### 3. Results
Results of rewards and episode lengths during evaluation will be auto-saved under ./log, but we manually rename and relocate the logs to their corresponding experiment. Code in result.ipynb is used to load and visualize the .npz logs.
#### 4. Utils
To shift between MLP and CNN policy, please uncomment the marked code piece in utils.py.
