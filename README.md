# Residual_Steps_Network

## Introduction
This is a pytorch realization of MSPN proposed in [ Residual Steps Network for Multi-Person Pose Estimation ][1]. which wins 2019 COCO Keypoints Challenge. The original repo is based on the inner deep learning framework (MegBrain) in Megvii Inc. 

In this work, we propose a novel network structure called Residual Steps Network (RSN) aiming to aggregate features inside each level (we define consecutive feature maps with the same spatial size as one level) of the network. RSN fuses the intra-level features to obtain better low-level delicate spatial information resulting in more precise keypoint localization. The proposed method outperforms the winner of COCO Keypoint Challenge 2018 and achieves state-of-the-art results on both COCO and MPII benchmarks, without using extra training data andpretrained model. Our single model achieves 78.6 on COCO test-dev, 93.0 on MPII test dataset. Ensembled models achieve 79.2 on COCO test-dev, 77.1 on COCO test-challenge. The source code will be publicly available for further research.

![Overview of RSN.](/figures/RSN.png)


## Results

### Results on COCO val dataset
| Model | Input Size | GFLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | AR | 
| :-----------------: | :-----------: | :--------: | :------: |:------: | :------: | :------: | :------: | :------: |
| RSN-18 | 256x192 | 2.5 | 71.5 | 90.1 | 78.4 | 67.4 | 77.5 | 77.0 | 93.2 | 83.1 | 72.6 | 83.1 |
| RSN-50 | 256x192 | 6.4 | 74.5 | 91.2 | 81.2 | 70.5 | 80.4 | 79.7 | 94.2 | 85.6 | 75.4 | 85.7 |
| RSN-101 | 256x192 | 11 | 75.2 | 91.5 | 82.2 | 71.1 | 81.1 | 80.3 | 94.3 | 86.4 | 76.0 | 86.4 |
| 2×RSN-50 | 256x192 | 13.9 |  77.1 | 91.8 | 82.9 | 72.0 | 81.6 | 81.1 | 94.9 | 87.1 | 76.9 | 87.0 |
| 3×RSN-50 | 256x192 |  20  | 76.9 | 91.8 | 83.2 | 72.7 | 83.1 | 81.8 | 94.8 | 87.3 | 77.4 | 87.8 |
| 4×RSN-50 | 256x192 | 31.5 | 76.9 | 91.8 | 83.2 | 72.7 | 83.1 | 81.8 | 94.8 | 87.3 | 77.4 | 87.8 |
| 4×RSN-50 | 384x288 | 70.9 | 76.9 | 91.8 | 83.2 | 72.7 | 83.1 | 81.8 | 94.8 | 87.3 | 77.4 | 87.8 |


### Results on COCO test-dev dataset
| Model | Input Size | GFLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | AR |
| :-----------------: | :-----------: | :--------: | :------: | :------: | :------: | :------: | :------: | :------: |
| RSN-18 | 256x192 | 2.5 | 71.5 | 90.1 | 78.4 | 67.4 | 77.5 | 77.0 | 93.2 | 83.1 | 72.6 | 83.1 |
| RSN-50 | 256x192 | 6.4 | 74.5 | 91.2 | 81.2 | 70.5 | 80.4 | 79.7 | 94.2 | 85.6 | 75.4 | 85.7 |
| 2×RSN-50 | 256x192 | 13.9 |  77.1 | 91.8 | 82.9 | 72.0 | 81.6 | 81.1 | 94.9 | 87.1 | 76.9 | 87.0 |
| 4×RSN-50 | 256x192 | 31.5 | 76.9 | 91.8 | 83.2 | 72.7 | 83.1 | 81.8 | 94.8 | 87.3 | 77.4 | 87.8 |
| 4×RSN-50 | 384x288 | 70.9 | 76.9 | 91.8 | 83.2 | 72.7 | 83.1 | 81.8 | 94.8 | 87.3 | 77.4 | 87.8 |
| 4×RSN-50<sup>\+</sup> | - | - | 78.1 | 94.1 | 85.9 | 74.5 | 83.3 | 83.1 | 96.7 | 89.8 | 79.3 | 88.2 |

### Results on MPII dataset
| Model | Split | Input Size | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean |
| :-----------------: | :------------------: | :-----------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| 4-stg MSPN | val | 256x256 | 96.8 | 96.5 | 92.0 | 87.0 | 89.9 | 88.0 | 84.0 | 91.1 |
| 4-stg MSPN | test | 256x256 | 98.4 | 97.1 | 93.2 | 89.2 | 92.0 | 90.1 | 85.5 | 92.6 |

#### Note
* \+ means using model ensemble.
