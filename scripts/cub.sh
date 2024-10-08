#!/usr/bin/bash
set -e  # Exit immediately if any command fails

# 22 concepts
python train.py -c /notebooks/AR-Imperial-Thesis/configs/CBM/Independent/DT_as_label_encoder/cub_reduced_pretrained_xtoc_1.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/CBM/Sequential/DT_as_label_encoder/cub_reduced_pretrained_xtoc_1.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/MCBM_Seq/cub_reduced_pretrained_xtoc_1.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/MCBM_Joint/cub_reduced_pretrained_xtoc_1.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth

# 29 concepts
python train.py -c /notebooks/AR-Imperial-Thesis/configs/CBM/Independent/DT_as_label_encoder/cub_reduced_pretrained_xtoc_2.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/CBM/Sequential/DT_as_label_encoder/cub_reduced_pretrained_xtoc_2.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/MCBM_Seq/cub_reduced_pretrained_xtoc_2.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/MCBM_Joint/cub_reduced_pretrained_xtoc_2.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth

# 45 concepts
python train.py -c /notebooks/AR-Imperial-Thesis/configs/CBM/Independent/DT_as_label_encoder/cub_reduced_pretrained_xtoc_3.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/CBM/Sequential/DT_as_label_encoder/cub_reduced_pretrained_xtoc_3.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/MCBM_Seq/cub_reduced_pretrained_xtoc_3.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/MCBM_Joint/cub_reduced_pretrained_xtoc_3.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth

# 59 concepts
python train.py -c /notebooks/AR-Imperial-Thesis/configs/CBM/Independent/DT_as_label_encoder/cub_reduced_pretrained_xtoc_4.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/CBM/Sequential/DT_as_label_encoder/cub_reduced_pretrained_xtoc_4.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/MCBM_Seq/cub_reduced_pretrained_xtoc_4.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/MCBM_Joint/cub_reduced_pretrained_xtoc_4.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth

# 112 concepts
python train.py -c /notebooks/AR-Imperial-Thesis/configs/CBM/Independent/DT_as_label_encoder/cub_pretrained_xtoc.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/CBM/Sequential/DT_as_label_encoder/cub_pretrained_xtoc.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/MCBM_Seq/cub_pretrained_xtoc.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/MCBM_Joint/cub_pretrained_xtoc.json --sd saved_cub --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth

# baselines
python train.py -c /notebooks/AR-Imperial-Thesis/configs/CBM/Independent/cub_reduced_pretrained_xtoc_3.json --sd saved_cub_hard_entropy --entropy_layer 'true' --tau 0.6 --lm 0.00001 --hard_cbm 0 --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/CBM/Independent/cub_reduced_pretrained_xtoc_3.json --sd saved_cub_indep_entropy --entropy_layer 'true' --tau 0.6 --lm 0.00001 --hard_cbm 1 --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth
python train.py -c /notebooks/AR-Imperial-Thesis/configs/CBM/Sequential/cub_reduced_pretrained_xtoc_3.json --sd saved_cub_seq_entropy --entropy_layer 'true' --tau 0.6 --lm 0.00001 --pretrained_concept_predictor /notebooks/AR-Imperial-Thesis/model_best.pth