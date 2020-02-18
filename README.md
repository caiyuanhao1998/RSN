# Residual_Steps_Network

## Introduction
This is a pytorch realization of RSN proposed in [ Residual Steps Network for Multi-Person Pose Estimation ][1]. which wins 2019 COCO Keypoints Challenge. The original repo is based on the inner deep learning framework (MegBrain) in Megvii Inc. 

In this work, we propose a novel network structure called Residual Steps Network (RSN) aiming to aggregate features inside each level (we define consecutive feature maps with the same spatial size as one level) of the network. RSN fuses the intra-level features to obtain better low-level delicate spatial information resulting in more precise keypoint localization. The proposed method outperforms the winner of COCO Keypoint Challenge 2018 and achieves state-of-the-art results on both COCO and MPII benchmarks, without using extra training data andpretrained model. Our single model achieves 78.6 on COCO test-dev, 93.0 on MPII test dataset. Ensembled models achieve 79.2 on COCO test-dev, 77.1 on COCO test-challenge. The source code is publicly available for further research.

![Overview of RSN.](/figures/pipeline_v2.png)
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

### Results on MPII dataset
| Model | Split | Input Size | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean |
| :-----------------: | :------------------: | :-----------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| 4×RSN-50 | val | 256x256 | 96.7 | 96.7 | 92.3 | 88.2 | 90.3 | 89.0 | 85.3 | 91.6 |
| 4×RSN-50 | test | 256x256 | 98.5 | 97.3 | 93.9 | 89.9 | 92.0 | 90.6 | 86.8 | 93.0 |

#### Note
* \+ means using model ensemble.
