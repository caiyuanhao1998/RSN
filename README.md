	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-delicate-local-representations-for/keypoint-detection-on-coco)](https://paperswithcode.com/sota/keypoint-detection-on-coco?p=learning-delicate-local-representations-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-delicate-local-representations-for/keypoint-detection-on-coco-test-challenge)](https://paperswithcode.com/sota/keypoint-detection-on-coco-test-challenge?p=learning-delicate-local-representations-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-delicate-local-representations-for/multi-person-pose-estimation-on-coco)](https://paperswithcode.com/sota/multi-person-pose-estimation-on-coco?p=learning-delicate-local-representations-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-delicate-local-representations-for/pose-estimation-on-coco-test-dev)](https://paperswithcode.com/sota/pose-estimation-on-coco-test-dev?p=learning-delicate-local-representations-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-delicate-local-representations-for/pose-estimation-on-mpii-single-person)](https://paperswithcode.com/sota/pose-estimation-on-mpii-single-person?p=learning-delicate-local-representations-for)

# Learning Delicate Local Representations for Multi-Person Pose Estimation (ECCV 2020 Spotlight)
[![winner](https://img.shields.io/badge/Winner-COCO_2019_Keypoint_Challenge-179bd3)](http://cocodataset.org/#keypoints-leaderboard)
[![bpa](https://img.shields.io/badge/COCO-Best_Paper_Award-179bd3)](https://github.com/caiyuanhao1998/RSN/blob/master/figures/2019_best_paper.png)
[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2003.04030)
[![zhihu](https://img.shields.io/badge/zhihu-知乎中文解读-179bd3)](https://zhuanlan.zhihu.com/p/112297707)
![visitors](https://visitor-badge.glitch.me/badge?page_id=caiyuanhao1998/RSN)

*This is a pytorch realization of [Residual Steps Network][11] **which won 2019 COCO Keypoint Challenge and ranks 1st place on both COCO test-dev and test-challenge datasets as shown in [COCO leaderboard][1]**. The original repo is based on the inner deep learning framework (MegBrain) in Megvii Inc.*

#### News
- **2020.07 :** Our paper has been accepted as Spotlight by ECCV 2020 :rocket: 
- **2019.09 :** Our work won the **First place** and **Best Paper Award** in COCO 2019 Keypoint Challenge :trophy: 


<hr />

> **Abstract:** *In this paper, we propose a novel method called Residual Steps Network (RSN). RSN aggregates features with the same spatialsize (Intra-level features) efficiently to obtain delicate local representations, which retain rich low-level spatial information and result in precise keypoint localization. In addition, we propose an efficient attention mechanism - Pose Refine Machine (PRM) to further refine the keypoint locations. Our approach won the 1st place of COCO Keypoint Challenge 2019 and achieves state-of-the-art results on both COCO and MPII benchmarks, without using extra training data and pretrained model. Our single model achieves 78.6 on COCO test-dev, 93.0 on MPII test dataset. Ensembled models achieve 79.2 on COCO test-dev, 77.1 on COCO test-challenge dataset.*
<hr />

## Pipieline of Residual Steps Network
![Overview of RSN.](/figures/pipeline_v2.png)

## Architecture of Pose Refine Machine
![Overview of RSN.](/figures/RM.png)


## Some prediction resullts of our method on COCO and MPII valid datasets
![Prediction Results of COCO-valid.](/figures/results.png)

![Prediction Results of MPII-valid.](/figures/results_mpii.png)


## Results(MegDL Version)

### Results on COCO val dataset
| Model | Input Size | GFLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | AR | 
| :-----------------: | :-----------: | :--------: | :------: |:------: | :------: | :------: | :------: | :------: |
| Res-18 | 256x192 | 2.3 | 70.7 | 89.5 | 77.5 | 66.8 | 75.9 | 75.8 | 92.8 | 83.2 | 72.7 | 82.1 |
| RSN-18 | 256x192 | 2.5 | 73.6 | 90.5 | 80.9 | 67.8 | 79.1 | 78.8 | 93.7 | 85.2 | 74.7 | 84.5 |
| RSN-50 | 256x192 | 6.4 | 74.7 | 91.4 | 81.5 | 71.0 | 80.2 | 80.0 | 94.4 | 86.2 | 76.0 | 85.7 |
| RSN-101 | 256x192 | 11.5 | 75.8 | 92.4 | 83.0 | 72.1 | 81.2 | 81.1 | 95.6 | 87.6 | 77.2 | 86.5 |
| 2×RSN-50 | 256x192 | 13.9 | 77.2 | 92.3 | 84.0 | 73.8 | 82.5 | 82.2 | 95.1 | 88.0 | 78.4 | 87.5 |
| 3×RSN-50 | 256x192 | 20.7 | 78.2 | 92.3 | 85.1 | 74.7 | 83.7 | 83.1 | 95.9 | 89.1 | 79.3 | 88.5 |
| 4×RSN-50 | 256x192 | 29.3 | 79.0 | 92.5 | 85.7 | 75.2 | 84.5 | 83.7 | 95.5 | 89.4 | 79.8 | 89.0 |
| 4×RSN-50 | 384x288 | 65.9 | 79.6 | 92.5 | 85.8 | 75.5 | 85.2 | 84.2 | 95.6 | 89.8 | 80.4 | 89.5 |


### Results on COCO test-dev dataset
| Model | Input Size | GFLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | AR |
| :-----------------: | :-----------: | :--------: | :------: | :------: | :------: | :------: | :------: | :------: |
| RSN-18 | 256x192 | 2.5 | 71.6 | 92.6 | 80.3 | 68.8 | 75.8 | 77.7 |
| RSN-50 | 256x192 | 6.4 | 72.5 | 93.0 | 81.3 | 69.9 | 76.5 | 78.8 |
| 2×RSN-50 | 256x192 | 13.9 |  75.5 | 93.6 | 84.0 | 73.0 | 79.6 | 81.3 |
| 4×RSN-50 | 256x192 | 29.3 | 78.0 | 94.2 | 86.5 | 75.3 | 82.2 | 83.4 |
| 4×RSN-50 | 384x288 | 65.9 | 78.6 | 94.3 | 86.6 | 75.5 | 83.3 | 83.8 |
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




## Results(Pytorch Version)
### Results on COCO val dataset
 Model | Input Size | GFLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | AR | 
| :-----------------: | :-----------: | :--------: | :------: |:------: | :------: | :------: | :------: | :------: |
| Res-18 | 256x192 | 2.3 | 65.2 | 87.3 | 71.5 | 61.2 | 72.2 | 71.3 | 91.4 | 77.0 | 68.7 | 79.5 |
| RSN-18 | 256x192 | 2.5 | 70.4 | 88.8 | 77.7 | 67.2 | 76.7 | 76.5 | 93.0 | 82.8 | 72.2 | 82.5 |


#### Note
* \+ means using ensemble models.
* All models are trained on 8 V100 GPUs
* We done all the experiments using our own DL-Platform MegDL, all results in our paper are reported on MegDL. There are some differences between MegDL and Pytorch. MegDL will be released in March. The MegDL code and model will be also publicly avaible.

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

2. Download ground truth from [Google Drive][6] or [Baidu Drive][10] (code: fc51), and put it into **$RSN_HOME/dataset/COCO/gt_json/**.

3. Download detection result from [Google Drive][6] or [Baidu Drive][10] (code: fc51), and put it into **$RSN_HOME/dataset/COCO/det_json/**.

#### MPII

1. Download images from [MPII website][5], and put images into **$RSN_HOME/dataset/MPII/images/**.

2. Download ground truth from [Google Drive][6] or [Baidu Drive][10] (code: fc51), and put it into **$RSN_HOME/dataset/MPII/gt_json/**.

3. Download detection result from [Google Drive][6] or [Baidu Drive][10] (code: fc51), and put it into **$RSN_HOME/dataset/MPII/det_json/**.


### Log
Create a directory to save logs and models:
```
mkdir $RSN_HOME/model_logs
```

### Train
Go to specified experiment repository, e.g.
```
cd $RSN_HOME/exps/RSN50.coco
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
@inproceedings{cai2020learning,
  title={Learning Delicate Local Representations for Multi-Person Pose Estimation},
  author={Yuanhao Cai and Zhicheng Wang and Zhengxiong Luo and Binyi Yin and Angang Du and Haoqian Wang and Xinyu Zhou and Erjin Zhou and Xiangyu Zhang and Jian Sun},
  booktitle={ECCV},
  year={2020}
}

@inproceedings{cai2019res,
  title={Res-steps-net for multi-person pose estimation},
  author={Cai, Yuanhao and Wang, Zhicheng and Yin, Binyi and Yin, Ruihao and Du, Angang and Luo, Zhengxiong and Li, Zeming and Zhou, Xinyu and Yu, Gang and Zhou, Erjin and others},
  booktitle={Joint COCO and Mapillary Workshop at ICCV},
  year={2019}
}
```
And the [code][7] of [Cascaded Pyramid Network][8] is also available. 

## Contact
You can contact us by email published in our [paper][11].

[1]: http://cocodataset.org/#keypoints-leaderboard
[2]: https://pytorch.org/
[3]: https://github.com/cocodataset/cocoapi
[4]: http://cocodataset.org/#download
[5]: http://human-pose.mpi-inf.mpg.de/
[6]: https://drive.google.com/open?id=14zW0YZ0A9kPMNt_wjBpQZg5xBiW5ecPd
[7]: https://github.com/megvii-detection/tf-cpn
[8]: https://arxiv.org/abs/1711.07319
[9]: https://github.com/fenglinglwb/MSPN
[10]: https://pan.baidu.com/s/1MqpmR7EkZu3G_Hi0_4NFTA
[11]: https://arxiv.org/abs/2003.04030
