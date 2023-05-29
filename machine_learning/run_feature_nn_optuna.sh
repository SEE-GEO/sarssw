#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH -p chair
#SBATCH -A C3SE512-22-1
#SBATCH --gpus-per-node=A100:4
script_file="feature_nn_optuna.py"
echo "Starting run of ${script_file}"

cp /cephyr/users/${USER}/Alvis/sarssw/machine_learning/${script_file} ${TMPDIR}/script.py
cp /mimer/NOBACKUP/priv/chair/sarssw/apptainer_sarssw.sif ${TMPDIR}/apptainer.sif
echo "Copying data"
mkdir -p ${TMPDIR}/data
time cp -r /mimer/NOBACKUP/priv/chair/sarssw/IW_VV_VH_50k/train ${TMPDIR}/data/train
time cp -r /mimer/NOBACKUP/priv/chair/sarssw/IW_VV_VH_50k/val ${TMPDIR}/data/val

module purge
apptainer exec --nv ${TMPDIR}/apptainer.sif \
    python ${TMPDIR}/script.py \
    --data_dir ${TMPDIR}/data \
    --dataframe_path /mimer/NOBACKUP/priv/chair/sarssw/sar_dataset_features_labels_22_may/sar_dataset.pickle \
    --gpus 4
echo 'sbatch job done'