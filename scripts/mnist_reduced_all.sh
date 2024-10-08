#!/usr/bin/bash
set -e  # Exit immediately if any command fails

python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Independent/DT_as_label_encoder/mnist_pretrained_xtoc.json --sd saved_mnist
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Sequential/DT_as_label_encoder/mnist_pretrained_xtoc.json --sd saved_mnist
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Seq/mnist_pretrained_xtoc.json --sd saved_mnist
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mnist_pretrained_xtoc_joint.json --sd saved_mnist