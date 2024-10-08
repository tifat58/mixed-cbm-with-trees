#!/usr/bin/bash
set -e  # Exit immediately if any command fails

# msl=150
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Independent/DT_as_label_encoder/mnist_full_pretrained_xtoc.json --msl 150 --sd saved_mnist_full_msl_150
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Sequential/DT_as_label_encoder/mnist_full_pretrained_xtoc.json --msl 150 --sd saved_mnist_full_msl_150
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Seq/mnist_full_pretrained_xtoc.json --msl 150 --sd saved_mnist_full_msl_150
#
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mnist_full_pretrained_xtoc_joint.json --msl 150 --sd saved_mnist_full_msl_150  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mnist_full/alpha=0.1/0922_145308/model_best.pth"
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mnist_full_pretrained_xtoc_joint.json --msl 150 --sd saved_mnist_full_msl_150  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mnist_full/alpha=1.0/0922_142928/model_best.pth"
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mnist_full_pretrained_xtoc_joint.json --msl 150 --sd saved_mnist_full_msl_150  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mnist_full/alpha=100/0922_144309/model_best.pth"
#
## msl=30
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Independent/DT_as_label_encoder/mnist_full_pretrained_xtoc.json --msl 30 --sd saved_mnist_full_msl_30
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Sequential/DT_as_label_encoder/mnist_full_pretrained_xtoc.json --msl 30 --sd saved_mnist_full_msl_30
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Seq/mnist_full_pretrained_xtoc.json --msl 30 --sd saved_mnist_full_msl_30
#
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mnist_full_pretrained_xtoc_joint.json --msl 30 --sd saved_mnist_full_msl_30  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mnist_full/alpha=0.1/0922_145308/model_best.pth"
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mnist_full_pretrained_xtoc_joint.json --msl 30 --sd saved_mnist_full_msl_30  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mnist_full/alpha=1.0/0922_142928/model_best.pth"
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mnist_full_pretrained_xtoc_joint.json --msl 30 --sd saved_mnist_full_msl_30  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mnist_full/alpha=100/0922_144309/model_best.pth"
#
## msl=5
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Independent/DT_as_label_encoder/mnist_full_pretrained_xtoc.json --msl 5 --sd saved_mnist_full_msl_5
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Sequential/DT_as_label_encoder/mnist_full_pretrained_xtoc.json --msl 5 --sd saved_mnist_full_msl_5
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Seq/mnist_full_pretrained_xtoc.json --msl 5 --sd saved_mnist_full_msl_5
#
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mnist_full_pretrained_xtoc_joint.json --msl 5 --sd saved_mnist_full_msl_5  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mnist_full/alpha=0.1/0922_145308/model_best.pth"
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mnist_full_pretrained_xtoc_joint.json --msl 5 --sd saved_mnist_full_msl_5  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mnist_full/alpha=1.0/0922_142928/model_best.pth"
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mnist_full_pretrained_xtoc_joint.json --msl 5 --sd saved_mnist_full_msl_5  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mnist_full/alpha=100/0922_144309/model_best.pth"
#

# baselines
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Independent/mnist_full_pretrained_xtoc.json --sd saved_el_mnist_full_hard --hard_cbm 1 --entropy_layer 'true' --tau 0.6 --lm 0.0001
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Independent/mnist_full_pretrained_xtoc.json --sd saved_el_mnist_full_ind --hard_cbm 0 --entropy_layer 'true' --tau 0.6 --lm 0.0001
#python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Sequential/mnist_full_pretrained_xtoc.json --sd saved_el_mnist_full_seq --entropy_layer 'true' --tau 0.6 --lm 0.0001