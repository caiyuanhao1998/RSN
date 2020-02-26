# Residual Steps Network for Multi-Person Pose Estimation

## Introduction
This is a pytorch realization of Residual Steps Network which wins 2019 COCO Keypoints Challenge and ranks 1st place on both COCO test-dev and test-challenge datasets as shown in [COCO leaderboard][1]. The original repo is based on the inner deep learning framework (MegBrain) in Megvii Inc. 

In this paper, we propose a novel network structure called Residual Steps Network (RSN) aiming to aggregate features inside each
level (we define consecutive feature maps with the same spatial size as one level) of the network. RSN fuses the intra-level features to obtain better low-level delicate spatial information resulting in more precise keypoint localization. The proposed method outperforms the winner of COCO Keypoint Challenge 2018 and achieves state-of-the-art results on both COCO and MPII benchmarks, **without using extra training data and pretrained model**. Our single model achieves 78.6 on COCO test-dev, 93.0 on MPII test dataset. Ensembled models achieve 79.2 on COCO test-dev, 77.1 on COCO test-challenge. The source code is publicly available for further research.

## Pipieline of Multi-stage Residual Steps Network
![Overview of RSN.](/figures/pipeline_v2.png)


## Some prediction resullts of our method on COCO and MPII valid datasets
![Prediction Results of COCO-valid.](/figures/results.png)

![Prediction Results of MPII-valid.](/figures/results_mpii.png)


## Results

### Results on COCO val dataset
| Model | Input Size | GFLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | AR | 
| :-----------------: | :-----------: | :--------: | :------: |:------: | :------: | :------: | :------: | :------: |
| RSN-18 | 256x192 | 2.5 | 73.6 | 90.5 | 80.9 | 67.8 | 79.1 | 78.8 | 93.7 | 85.2 | 74.7 | 84.5 |
| RSN-50 | 256x192 | 6.4 | 74.7 | 91.4 | 81.5 | 71.0 | 80.2 | 80.0 | 94.4 | 86.2 | 76.0 | 85.7 |
| RSN-101 | 256x192 | 11.5 | 75.8 | 92.4 | 83.0 | 72.1 | 81.2 | 81.1 | 95.6 | 87.6 | 77.2 | 86.5 |
| 2×RSN-50 | 256x192 | 13.9 | 77.2 | 92.3 | 84.0 | 73.8 | 82.5 | 82.2 | 95.1 | 88.0 | 78.4 | 87.5 |
| 3×RSN-50 | 256x192 | 20.7 | 78.2 | 92.3 | 85.1 | 74.7 | 83.7 | 83.1 | 95.9 | 89.1 | 79.3 | 88.5 |
| 4×RSN-50 | 256x192 | 31.5 | 79.0 | 92.5 | 85.7 | 75.2 | 84.5 | 83.7 | 95.5 | 89.4 | 79.8 | 89.0 |
| 4×RSN-50 | 384x288 | 70.9 | 79.6 | 92.5 | 85.8 | 75.5 | 85.2 | 84.2 | 95.6 | 89.8 | 80.4 | 89.5 |


### Results on COCO test-dev dataset
| Model | Input Size | GFLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | AR |
| :-----------------: | :-----------: | :--------: | :------: | :------: | :------: | :------: | :------: | :------: |
| RSN-18 | 256x192 | 2.5 | 71.6 | 92.6 | 80.3 | 68.8 | 75.8 | 77.7 |
| RSN-50 | 256x192 | 6.4 | 72.5 | 93.0 | 81.3 | 69.9 | 76.5 | 78.8 |
| 2×RSN-50 | 256x192 | 13.9 |  75.5 | 93.6 | 84.0 | 73.0 | 79.6 | 81.3 |
| 4×RSN-50 | 256x192 | 31.5 | 78.0 | 94.2 | 86.5 | 75.3 | 82.2 | 83.4 |
| 4×RSN-50 | 384x288 | 70.9 | 78.6 | 94.3 | 86.6 | 75.5 | 83.3 | 83.8 |
| 4×RSN-50<sup>\+</sup> | - | - | 79.2 | 94.4 | 87.1 | 76.1 | 83.8 | 84.1 |

### Results on COCO test-challenge dataset
| Model | Input Size | GFLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | AR |
| :-----------------: | :-----------: | :--------: | :------: | :------: | :------: | :------: | :------: | :------: |
| 4×RSN-50<sup>\+</sup> | - | - | 77.1 | 93.3 | 83.6 | 72.2 | 83.6 | 82.6 |

### Results on MPII dataset
| Model | Split | Input Size | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean |
| :-----------------: | :------------------: | :-----------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| 4×RSN-50 | val | 256x256 | 96.7 | 96.7 | 92.3 | 88.2 | 90.3 | 89.0 | 85.3 | 91.6 |
| 4×RSN-50 | test | 256x256 | 98.5 | 97.3 | 93.9 | 89.9 | 92.0 | 90.6 | 86.8 | 93.0 |

#### Note
* \+ means using model ensemble.

## Repo Structure
This repo is organized as following:
```
$RSN_HOME
|-- cvpack
|
|-- dataset
|   |-- COCO
|   |   |-- det_json
|   |   |-- gt_json
|   |   |-- images
|   |       |-- train2014
|   |       |-- val2014
|   |
|   |-- MPII
|       |-- det_json
|       |-- gt_json
|       |-- images
|   
|-- lib
|   |-- models
|   |-- utils
|
|-- exps
|   |-- exp1
|   |-- exp2
|   |-- ...
|
|-- model_logs
|
|-- README.md
|-- requirements.txt
```

## Quick Start

### Installation

1. Install Pytorch referring to [Pytorch website][2].

2. Clone this repo, and config **RSN_HOME** in **/etc/profile** or **~/.bashrc**, e.g.
 ```
 export RSN_HOME='/path/of/your/cloned/repo'
 export PYTHONPATH=$PYTHONPATH:$RSN_HOME
 ```

3. Install requirements:
 ```
 pip3 install -r requirements.txt
 ```

4. Install COCOAPI referring to [cocoapi website][3], or:
 ```
 git clone https://github.com/cocodataset/cocoapi.git $RSN_HOME/lib/COCOAPI
 cd $RSN_HOME/lib/COCOAPI/PythonAPI
 make install
 ```
 
### Dataset

#### COCO

1. Download images from [COCO website][4], and put train2014/val2014 splits into **$RSN_HOME/dataset/COCO/images/** respectively.

2. Download ground truth from [Google Drive][6], and put it into **$RSN_HOME/dataset/COCO/gt_json/**.

3. Download detection result from [Google Drive][6], and put it into **$RSN_HOME/dataset/COCO/det_json/**.

#### MPII

1. Download images from [MPII website][5], and put images into **$RSN_HOME/dataset/MPII/images/**.

2. Download ground truth from [Google Drive][6], and put it into **$RSN_HOME/dataset/MPII/gt_json/**.

3. Download detection result from [Google Drive][6], and put it into **$RSN_HOME/dataset/MPII/det_json/**.

### Model
For your convenience, We provide well-trained RSN-18, RSN-50, 4×RSN-18 for COCO and RSN-18, RSN-50 for MPII.

### Log
Create a directory to save logs and models:
```
mkdir $RSN_HOME/model_logs
```

### Train
Go to specified experiment repository, e.g.
```
cd $RSN_HOME/exps/rsn.2xstg.coco
```
and run:
```
python config.py -log
python -m torch.distributed.launch --nproc_per_node=gpu_num train.py
```
the ***gpu_num*** is the number of gpus.

### Test
```
python -m torch.distributed.launch --nproc_per_node=gpu_num test.py -i iter_num
```
the ***gpu_num*** is the number of gpus, and ***iter_num*** is the iteration number you want to test.

## Citation
Please considering citing our projects in your publications if they help your research.
```
@article{li2019rethinking,
  title={Rethinking on Multi-Stage Networks for Human Pose Estimation},
  author={Li, Wenbo and Wang, Zhicheng and Yin, Binyi and Peng, Qixiang and Du, Yuming and Xiao, Tianzi and Yu, Gang and Lu, Hongtao and Wei, Yichen and Sun, Jian},
  journal={arXiv preprint arXiv:1901.00148},
  year={2019}
}

@inproceedings{chen2018cascaded,
  title={Cascaded pyramid network for multi-person pose estimation},
  author={Chen, Yilun and Wang, Zhicheng and Peng, Yuxiang and Zhang, Zhiqiang and Yu, Gang and Sun, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7103--7112},
  year={2018}
}
```
And the [code][7] of [Cascaded Pyramid Network][8] is also available. 

## Contact
You can contact us by email published in our [paper][1] or 3359145729@qq.com.

[1]: http://cocodataset.org/
[2]: https://pytorch.org/
[3]: https://github.com/cocodataset/cocoapi
[4]: http://cocodataset.org/#download
[5]: http://human-pose.mpi-inf.mpg.de/
[6]: https://drive.google.com/open?id=1MW27OY_4YetEZ4JiD4PltFGL_1-caECy
[7]: https://github.com/megvii-detection/tf-cpn
[8]: https://arxiv.org/abs/1711.07319
[9]: https://github.com/fenglinglwb/MSPN
