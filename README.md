# ADStereo

This paper presents two sampling strategies: the Adaptive Downsampling Module (ADM) and the Disparity Alignment Module (DAM), to prioritize real-time inference while ensuring accuracy. The ADM leverages local features to learn adaptive weights, enabling more effective downsampling while preserving crucial structure information. On the other hand, the DAM employs a learnable interpolation strategy to predict transformation offsets of pixels, thereby mitigating the spatial misalignment issue. 
Building upon these modules, we introduce **ADStereo**, a real-time yet accurate network that achieves highly competitive performance on multiple public benchmarks. 

# Demo on KITTI raw data
The pretrained KITTI model is loaded from  './fined/KITTI/' datafolders. 

https://github.com/user-attachments/assets/326230a6-871d-47ca-abf2-8a5ac4d959b7

Run `demo_video.py` to perform stereo matching on the raw Kitti sequence.
Here is an example result on our system with RTX a5000ada on Ubuntu 20.04


# Adaptive Downsampling Module \& Disparity Alignment Module

|||
|--|--|
| <img src="https://github.com/cocowy1/ADStereo/blob/main/figs/ADM.png"> | <img src="https://github.com/cocowy1/ADStereo/blob/main/figs/disparity_alignment.png"> |

# Overview
<img width="900" src="https://github.com/cocowy1/ADStereo/blob/main/figs/framework.png"/></div>

# New Added
We introduce a more lightweight model called  **ADStereo_fast** (highly competetive performance & faster speed), also included in this repo.

# Quantative Results
|             |                                        | \multicolumn{6}{c|} {KITTI 2012}  | \multicolumn{3}{c|}{KITTI 2015}   |                                   |                                   |
|-------------|----------------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|
|             | \multirow{1}{*}[5pt]{Method}           | 3-Noc(\%)                         | 3-All(\%)                         | 5-Noc(\%)                         | 5-All(\%)                         | Avg-Noc(\textit{px}) | Avg-All(\textit{px}) | D1-bg(\%)                         | D1-fg(\%)                         | D1-All(\%)                        | \tabincell{c}{Inference            |
| %           | MADNet           | $3.75$                            | $9.20$                            | $4.66$                            | $3.45$                            | $8.41$               | $4.27$               | ${-}$                             | $0.02$                            |
|             | HITNet      | $1.41$                            | $1.89$                            | $0.96$                            | $1.29$                            | $0.4$                | $0.5$                | $1.74$                            | $3.20$                            | $1.98$                            | $\mathbf{0.016}$                   | $3.2$                            |
|             | AANet+             | $1.55$                            | $2.04$                            | $0.98$                            | $1.30$                            | $0.4$                | $0.5$                | $1.65$                            | $3.96$                            | $2.03$                            | $0.043$                            | $\textcolor{blue}{\textbf{1.7}}$ |
|             | BGNet+         | $1.62$                            | $2.03$                            | $0.90$                            | $1.16$                            | $0.5$                | $0.6$                | $1.81$                            | $4.09$                            | $2.19$                            | $0.026$                            | $1.8$                            |
|             | FADNet         | $2.04$                            | $2.46$                            | $1.19$                            | $1.45$                            | $0.5$                | $0.6$                | $2.50$                            | $3.10$                            | $2.60$                            | $0.035$                            | $3.9$                            |
|             | IGEV-Stereo    | $\textbf{1.12}$                   | $\textbf{1.44}$                   | ${0.73}$                          | ${0.94}$                          | $\textbf{0.4}$       | $\textbf{0.4}$       | $\textcolor{blue}{\textbf{1.38}}$ | $\textbf{2.67}$                   | $\textbf{1.59}$                   | $0.18$                             | $1.7$                            |
|             | LEAStereo| $\textcolor{blue}{\textbf{1.13}}$ | $\textcolor{blue}{\textbf{1.45}}$ | $\textbf{0.67}$                   | $\textbf{0.88}$                   | $0.5$                | ${0.5}$              | ${1.40}$                          | $\textcolor{blue}{\textbf{2.91}}$ | $\textcolor{blue}{\textbf{1.65}}$ | $0.23$                             | $5.6$                            |
| %           | CREStereo      | ${1.14}$                          |                                   |
| %  ${1.46}$ | ${0.76}$                               | ${0.95}$                          | $0.4$                             | ${0.5}$                           | ${1.45}$                          | ${2.86}$             | ${1.69}$             | $0.41$                            | $4.8$                             | 2080Ti                            |
|             | ACVNet       | ${1.13}$                          | ${1.47}$                          | $\textcolor{blue}{\textbf{0.71}}$ | $\textcolor{blue}{\textbf{0.91}}$ | $0.4$                | ${0.5}$              | $\textbf{1.37}$                   | ${3.07}$                          | $\textcolor{blue}{\textbf{1.65}}$ | $0.20$                             | $5.2$                            |
|             | CFNet            | $1.23$                            | $1.58$                            | $0.74$                            | $0.94$                            | $0.4$                | $0.5$                | $1.54$                            | $3.56$                            | $1.88$                            | $0.14$                             | $3.8$                            |
|             | GANet-Deep         | $1.19$                            | $1.60$                            | $0.76$                            | $1.02$                            | $0.4$                | $0.5$                | $1.48$                            | $3.46$                            | $1.81$                            | $1.6$                              | $6.2$                            |
| %           | TBDN$       | N/A                               | N/A                               | N/A                               | N/A                               | N/A                  | N/A                  | $2.86$                            | N/A                               | $3.53$                            | $0.055$                            | $1.6$                            | 1060     |
|             | GwcNet           | $1.32$                            | $1.70$                            | $0.80$                            | $1.03$                            | $0.5$                | $0.5$                | $1.74$                            | $3.93$                            | $2.11$                            | $0.21$                             | $4.1$                            |
| %           | PSMNet        | $1.49$                            | $1.89$                            | $0.90$                            | $1.15$                            | $0.5$                | $0.6$                | $1.86$                            | $4.62$                            | $2.32$                            | $0.41$                             | $4.4$                            | Titan-Xp |
|             | \textbf{ADstereo (Ours)}               | ${1.36}$                          | ${1.68}$                          | ${0.83}$                          | ${1.04}$                          | $0.5$                | ${0.5}$              | ${1.59}$                          | ${2.94}$                          | ${1.82}$                          | $\textbf{\textcolor{blue}{0.054}}$ | \textbf{\textcolor{blue}{1.3}}$  |
|             | \textbf{ADstereo\_fast (Ours)}         | ${1.38}$                          | ${1.72}$                          | ${0.82}$                          | ${1.03}$                          | $0.5$                | ${0.5}$              | $1.57$                            | ${3.25}$                          | ${1.85}$                          | $\textbf{0.032}$                   | $\textbf{1.0}$                   |



# Model Zoo

All pretrained models are available in the [Google Driver:ADStereo](https://drive.google.com/drive/folders/1jdx4-gU8WuytiolZbGDLI-NSUHlQWuH4) and [Google Driver:ADStereo_fast](https://drive.google.com/drive/folders/1WcGgA7OS1lf5JJ3ajbXw-hMtz8cXrQ7k?dmr=1&ec=wgc-drive-globalnav-goto)

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
Run `main.py` to train on the SceneFlow dataset. Please update datapath in `main.py` as your training data path.

# 3. Finetune \& Inference 
Run `finetune.py` to finetune on the different real-world datasets, such as KITTI 2012, KITTI 2015, and ETH3D. Please update datapath in `finetune.py` as your training data path.

To generate prediction results on the test set of the KITTI dataset, you can run `evaluate_kitti.py`. 
The inference time can be printed  once you run `evaluate_kitti.py`. 
And the inference results on the KITTI dataset can be directly submitted to the online evaluation server for benchmarking.


# 4. Evaluate FLOPs 
Run `counts_op.py` to validate FLOPs consumption.

# 5. Results
<img width="1000" src="https://github.com/cocowy1/ADStereo/blob/main/figs/compare.png"/></div>


# Acknowledgements

This project is based on [GwcNet](https://github.com/xy-guo/GwcNet), [IGEV-Stereo](https://github.com/gangweiX/IGEV), and [CoEx](https://github.com/antabangun/coex). We thank the original authors for their excellent works.
