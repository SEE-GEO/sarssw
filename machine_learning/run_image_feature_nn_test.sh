#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH -p chair
#SBATCH -A C3SE512-22-1
#SBATCH --gpus-per-node=A100:4
cp /cephyr/users/${USER}/Alvis/sarssw/machine_learning/image_feature_nn_test.py ${TMPDIR}/script.py
cp /cephyr/users/${USER}/Alvis/sarssw/machine_learning/sarssw_ml_lib.py ${TMPDIR}/sarssw_ml_lib.py
cp /mimer/NOBACKUP/priv/chair/sarssw/apptainer_sarssw.sif ${TMPDIR}/apptainer.sif
echo "Copying data"
mkdir -p ${TMPDIR}/data
time cp -r /mimer/NOBACKUP/priv/chair/sarssw/IW_VV_VH_small/val ${TMPDIR}/data/val
time cp -r /mimer/NOBACKUP/priv/chair/sarssw/IW_VV_VH_small/test ${TMPDIR}/data/test
ls ${TMPDIR}
ls ${TMPDIR}/data

module purge
apptainer exec --nv ${TMPDIR}/apptainer.sif \
    python ${TMPDIR}/script.py \
    --data_dir ${TMPDIR}/data \
    --dataframe_path /mimer/NOBACKUP/priv/chair/sarssw/sar_dataset_features_labels_22_may/sar_dataset.pickle \
    --checkpoint "/cephyr/users/fborg/Alvis/sarssw/machine_learning/final_training_logger/lr=0.0005, dr=0.2, model=resnet18, pre=True/version_1/checkpoints/best_val_loss-epoch=08-val_loss=0.56.ckpt" \
    --gpus 4
echo 'sbatch job done'
