import sys
sys.path.append(".") 
import argparse
import json
import os
# import urllib.request
import numpy as np
import torch

from wilds import benchmark_datasets
from wilds import get_dataset
import matplotlib.pyplot as plt
import seaborn as sns



def main():
    correctness = np.load(os.path.join(args.trajectory, "correctness_1.npy"))
    group = np.load(os.path.join(args.weightpath, "group.npy"))

    plt.figure(figsize=(5.1, 3.6))
    for i in range(int(max(group))+1):
        g = (group==i)
        num = np.sum(g)
        correctness_g = correctness[g]
        # correctness_g = 1 - np.mean(correctness_g[:, 0:300], axis=1)
        correctness_g = 1 - np.mean(correctness_g[:, 1:11], axis=1)
        sns.kdeplot(correctness_g, bw=0.2, clip=[0, 1], shade=True, shade_lowest=False, legend=True, label='Group '+str(i)+ ' size: ' + str(num), linewidth=2)

        
    plt.xlabel("Uncertainty", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.xticks(np.arange(0, 1.1, step=0.2), fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, loc=1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'figure', 'correctness_1_group.png'))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions for WILDS datasets."
    )
    parser.add_argument(
        "--trajectorypath",
        type=str,
        help="Path to prediction trajectory.",
        default='',
    )

    # Parse args and run this script
    args = parser.parse_args()
    main()
