# UMIX: Improving Importance Weighting for Subpopulation Shift via Uncertainty-Aware Mixup

This repository contains the code of our paper [UMIX: Improving Importance Weighting for Subpopulation Shift via Uncertainty-Aware Mixup](https://arxiv.org/abs/2209.08928). The code is implemented on the code provided by WILDs. 
If you have any questions, **please contact me via the following email zongbo at tju.edu.cn**.

## Requirment

* Python 3
* torch 1.7.0
* torch-scatter 2.0.5
* torch-geometric 1.6.2
* torchvision 0.8.1+cu101
* torch-cluster 1.5.8
* torch-sparse 0.6.8  
* numpy 1.18.5
* pandas 1.1.5
* pillow 7.2.0
* scikit-learn 0.23.2
* scipy 1.5.2  
* transformers 4.15.0

## datasets

* waterbirds
* CelebA
* civilcomments
* camelyon17

## Usage

You can run our algorithm from the shell file in the script directory. 
Specifically, taking the Waterbirds dataset as an example, 
* you should first run the files in the UMIX_trajectory folder to get the uncertainty. 
* Then run the files in the UMIX folder to get the model results.
* You can then re-search for hyperparameters through the files under the UMIX_nni folder.

We also provide our saved checkpoints in this link.

## Acknowledgements
The code is built on WILDS codebase v1.2.2 (https://github.com/p-lambda/wilds/releases/tag/v1.2.2). 
We thank the builders of the original repository.
Because our method requires some adjustments to the original codebase, we directly upload our modified files for efficiency.

## Disclaimer

This is not an official Tencent product.


## Coypright

This tool is developed in Tencent AI Lab.

The copyright holder for this project is Tencent AI Lab.

All rights reserved.
