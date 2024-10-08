#!/usr/bin/bash
set -e  # Exit immediately if any command fails

#msl=150
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Independent/DT_as_label_encoder/mimic_pretrained_xtoc.json --msl 150 --sd saved_mimic_msl_150
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Sequential/DT_as_label_encoder/mimic_pretrained_xtoc.json --msl 150 --sd saved_mimic_msl_150
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Seq/mimic_pretrained_xtoc.json --msl 150 --sd saved_mimic_msl_150

python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mimic_pretrained_xtoc.json --msl 150 --sd saved_mimic_msl_150  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mimic/alpha=0.1/0928_232004/model_best.pth"
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mimic_pretrained_xtoc.json --msl 150 --sd saved_mimic_msl_150  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mimic/alpha=1/0928_231342/model_best.pth"
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mimic_pretrained_xtoc.json --msl 150 --sd saved_mimic_msl_150  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mimic/alpha=100/0928_232317/model_best.pth"

# msl=30
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Independent/DT_as_label_encoder/mimic_pretrained_xtoc.json --msl 30 --sd saved_mimic_msl_30
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Sequential/DT_as_label_encoder/mimic_pretrained_xtoc.json --msl 30 --sd saved_mimic_msl_30
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Seq/mimic_pretrained_xtoc.json --msl 30 --sd saved_mimic_msl_30

python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mimic_pretrained_xtoc.json --msl 30 --sd saved_mimic_msl_30  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mimic/alpha=0.1/0928_232004/model_best.pth"
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mimic_pretrained_xtoc.json --msl 30 --sd saved_mimic_msl_30  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mimic/alpha=1/0928_231342/model_best.pth"
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mimic_pretrained_xtoc.json --msl 30 --sd saved_mimic_msl_30  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mimic/alpha=100/0928_232317/model_best.pth"

# msl=5
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Independent/DT_as_label_encoder/mimic_pretrained_xtoc.json --msl 5 --sd saved_mimic_msl_5
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Sequential/DT_as_label_encoder/mimic_pretrained_xtoc.json --msl 5 --sd saved_mimic_msl_5
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Seq/mimic_pretrained_xtoc.json --msl 5 --sd saved_mimic_msl_5

python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mimic_pretrained_xtoc.json --msl 5 --sd saved_mimic_msl_5  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mimic/alpha=0.1/0928_232004/model_best.pth"
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mimic_pretrained_xtoc.json --msl 5 --sd saved_mimic_msl_5  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mimic/alpha=1/0928_231342/model_best.pth"
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/MCBM_Joint/mimic_pretrained_xtoc.json --msl 5 --sd saved_mimic_msl_5  --pretrained_concept_predictor_joint "/Users/gouse/PycharmProjects/mixed_cbms_with_trees/trained_models/pretrained_joined_models/joint_cbm_mimic/alpha=100/0928_232317/model_best.pth"

# baselines
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Independent/mimic_pretrained_xtoc_binary.json --sd saved_mimic_entropy_hard --hard_cbm 1 --entropy_layer 'true' --tau 0.6 --lm 0.0001
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Independent/mimic_pretrained_xtoc_binary.json --sd saved_mimic_entropy_ind --hard_cbm 0 --entropy_layer 'true' --tau 0.6 --lm 0.0001
python train.py -c /Users/gouse/PycharmProjects/mixed_cbms_with_trees/configs/CBM/Sequential/mimic_pretrained_xtoc_binary.json --sd saved_mimic_entropy_seq --entropy_layer 'true' --tau 0.6 --lm 0.0001