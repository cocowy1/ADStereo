# ADStereo

this paper presents two sampling strategies: the Adaptive Downsampling Module (ADM) and the Disparity Alignment Module (DAM), to prioritize real-time inference while ensuring accuracy. The ADM leverages local features to learn adaptive weights, enabling more effective downsampling while preserving crucial structure information. On the other hand, the DAM employs a learnable interpolation strategy to predict transformation offsets of pixels, thereby mitigating the spatial misalignment issue. 
Building upon these modules, we introduce **ADStereo**, a real-time yet accurate network that achieves highly competitive performance on multiple public benchmarks. 

# Demo on KITTI raw data
The pretrained KITTI model is loaded from  './fined/KITTI/' datafolders. 

Run `demo_video.py` to perform stereo matching on the raw Kitti sequence.
Here is an example result on our system with RTX a5000ada on Ubuntu 20.04

https://github.com/user-attachments/assets/0b06b06a-0033-4d5d-ac05-1bcca2b77675


# Adaptive Downsampling Module \& Disparity Alignment Module

|||
|--|--|
| <img src="https://github.com/cocowy1/ADStereo/blob/main/figs/ADM.png"> | <img src="https://github.com/cocowy1/ADStereo/blob/main/figs/disparity_alignment.png"> |

# Overview
<img width="900" src="https://github.com/cocowy1/ADStereo/blob/main/figs/framework.png"/></div>

The implementation of the improved version **ADStereo_plus** (stronger performance & faster speed) is also included in this repo.

# Model Zoo

All pretrained models are available in the [Google Drive](https://drive.google.com/drive/folders/1jdx4-gU8WuytiolZbGDLI-NSUHlQWuH4)

We assume the downloaded weights are located under the `./trained` directory. 

Otherwise, you may need to change the corresponding paths in the scripts.


# Environment
```
Python 3.9
Pytorch 2.4.0
```
# Create a virtual environment and activate it.
```
conda create -n ADStereo python=3.9
conda activate ADStereo
```

# Dependencies
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
pip install chardet
pip install imageio
pip install thop
pip install timm==0.5.4
```

# 1. Prepare training data
To evaluate/train ADStereo, you will need to download the required datasets.

[SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

[KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

[Middlebury](https://vision.middlebury.edu/stereo/submit3/)

[ETH3D](https://www.eth3d.net/datasets#low-res-two-view-test-data)

By default `datasets.py` will search for the datasets in these locations.

```bash
DATA
├── KITTI
│   ├── kitti_2012
│   │   └── training
        └── testing
│   ├── kitti_2015
│   │   └── training
        └── testing
└── SceneFlow
    ├── Driving
    │   ├── disparity
    │   └── frames_finalpass
    ├── FlyingThings3D
    │   ├── disparity
    │   └── frames_finalpass
    └── Monkaa
        ├── disparity
        └── frames_finalpass
└── Middlebury
    ├── trainingH
    ├── trainingH_GT
└── ETH3D
    ├── two_view_training
    ├── two_view_training_gt
```

# 2. Train on SceneFlow
Run `main.py` to train on the SceneFlow dataset. Please update datapath in `train_kitti.py` as your training data path.

# 3. Finetune on KITTI 2015 \& Inference 
Run `train_kitti.py` to finetune on the KITTI 2012 \& 2015 dataset. Please update datapath in `train_kitti.py` as your training data path.

To generate prediction results on the test set of the KITTI dataset, you can run `evaluate_kitti.py`. 
The inference time can be printed  once you run `evaluate_kitti.py`. 
And the inference results on the KITTI dataset can be directly submitted to the online evaluation server for benchmarking.

# 4. Finetune on Middlebury
Run `train_mid.py` to finetune on the KITTI 2012 \& 2015 dataset. Please update datapath in `train_mid.py` as your training data path.

# 5. Evaluate FLOPs 
Run `counts_op.py` to validate FLOPs consumption.

# 6. Results
<img width="1000" src="https://github.com/cocowy1/ADStereo/blob/main/figs/compare.png"/></div>
