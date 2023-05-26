#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH -p chair
#SBATCH -A C3SE512-22-1
#SBATCH --gpus-per-node=A40:1
cp /cephyr/users/brobeck/Alvis/sarssw/machine_learning/image_feature_nn_optuna.py ${TMPDIR}/script.py
cp /cephyr/users/brobeck/Alvis/sarssw/sandbox/apptainer_sarssw.sif ${TMPDIR}/apptainer.sif
echo "Copying data"
mkdir -p ${TMPDIR}/data
time cp -r /mimer/NOBACKUP/priv/chair/sarssw/IW_VV_VH_small/train ${TMPDIR}/data/train
time cp -r /mimer/NOBACKUP/priv/chair/sarssw/IW_VV_VH_small/val ${TMPDIR}/data/val
ls ${TMPDIR}
ls ${TMPDIR}/data

module purge
apptainer exec --nv ${TMPDIR}/apptainer.sif python ${TMPDIR}/script.py --data_dir ${TMPDIR}/data --dataframe_path /mimer/NOBACKUP/priv/chair/sarssw/sar_dataset_features_labels_22_may/sar_dataset.pickle --gpus 1
echo 'sbatch job done'
