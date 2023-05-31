#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH -p chair
#SBATCH -A C3SE512-22-1
#SBATCH --gpus-per-node=A100:4
echo 'running feature_nn_train.py'
cp /cephyr/users/${USER}/Alvis/sarssw/machine_learning/feature_nn_train.py ${TMPDIR}/script.py
cp /cephyr/users/${USER}/Alvis/sarssw/machine_learning/sarssw_ml_lib.py ${TMPDIR}/sarssw_ml_lib.py
cp /mimer/NOBACKUP/priv/chair/sarssw/apptainer_sarssw.sif ${TMPDIR}/apptainer.sif

module purge
apptainer exec --nv ${TMPDIR}/apptainer.sif \
    python ${TMPDIR}/script.py \
    --data_dir /mimer/NOBACKUP/priv/chair/sarssw/IW_VV_VH/ \
    --dataframe_path /mimer/NOBACKUP/priv/chair/sarssw/sar_dataset_features_labels_22_may/sar_dataset.pickle \
    --gpus 4
echo 'sbatch job done'
