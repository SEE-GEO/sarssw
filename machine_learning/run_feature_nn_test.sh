#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH -p chair
#SBATCH -A C3SE512-22-1
#SBATCH --gpus-per-node=A100:1
cp /cephyr/users/${USER}/Alvis/sarssw/machine_learning/feature_nn_test.py ${TMPDIR}/script.py
cp /cephyr/users/${USER}/Alvis/sarssw/machine_learning/sarssw_ml_lib.py ${TMPDIR}/sarssw_ml_lib.py

module purge
apptainer exec --nv /mimer/NOBACKUP/priv/chair/sarssw/apptainer_sarssw.sif  \
    python ${TMPDIR}/script.py \
    --data_dir /mimer/NOBACKUP/priv/chair/sarssw/IW_VV_VH \
    --dataframe_path /mimer/NOBACKUP/priv/chair/sarssw/sar_dataset_features_labels_22_may/sar_dataset.pickle \
    --checkpoint "/mimer/NOBACKUP/priv/chair/sarssw/final_only_feat/version_1/checkpoints/best_val_loss-epoch=41-val_loss=0.31.ckpt" \
    --save_dir "/mimer/NOBACKUP/priv/chair/sarssw/result_predictions_only_features" \
    --gpus 1

module purge
apptainer exec --nv /mimer/NOBACKUP/priv/chair/sarssw/apptainer_sarssw.sif  \
    python ${TMPDIR}/script.py \
    --data_dir /mimer/NOBACKUP/priv/chair/sarssw/IW_VV_VH \
    --dataframe_path /mimer/NOBACKUP/priv/chair/sarssw/sar_dataset_features_labels_22_may/sar_dataset.pickle \
    --checkpoint "/mimer/NOBACKUP/priv/chair/sarssw/final_only_feat/version_1/checkpoints/best_val_loss-epoch=55-val_loss=0.31.ckpt" \
    --save_dir "/mimer/NOBACKUP/priv/chair/sarssw/result_predictions_only_features" \
    --gpus 1

module purge
apptainer exec --nv /mimer/NOBACKUP/priv/chair/sarssw/apptainer_sarssw.sif  \
    python ${TMPDIR}/script.py \
    --data_dir /mimer/NOBACKUP/priv/chair/sarssw/IW_VV_VH \
    --dataframe_path /mimer/NOBACKUP/priv/chair/sarssw/sar_dataset_features_labels_22_may/sar_dataset.pickle \
    --checkpoint "/mimer/NOBACKUP/priv/chair/sarssw/final_only_feat/version_1/checkpoints/latest-epoch-epoch=99.ckpt" \
    --save_dir "/mimer/NOBACKUP/priv/chair/sarssw/result_predictions_only_features" \
    --gpus 1

echo 'sbatch job done'