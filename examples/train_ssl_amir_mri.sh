#!/bin/bash
# train a model to segment abdominal MRI (T2 fold of CHAOS challenge)
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1   # 0,1,2,3

####### Shared configs
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="myexperiments"
DATASET='AMIR_Superpix'
NWORKER=4

ALL_EV=( 0) # 5-fold cross validation (0, 1, 2, 3, 4)
ALL_SCALE=( "MIDDLE") # config of pseudolabels


### Exclude following classes from trianing
EXCLU='[2,3]' # setting 2: excluding cement paste and dark micro training set to test generalization capability even though they are unlabeled.
# EXCLU='[1,4]' # setting 2: excluding pores and bright micro training set to test generalization capability even though they are unlabeled.
# EXCLU='[3,4]' # setting 2: excluding bright and dark micros training set to test generalization capability even though they are unlabeled.
# EXCLU='[]' # setting 1: do not exclude anything (by Roy et al.)

### Testing classes
LABEL_SETS=0  # Use pores and bright micro as testing classes
#  LABEL_SETS=1 # Use cement past and dark micro as testing classes
#  LABEL_SETS=2 # Use bright and dark micros as testing classes

###### Training configs ######
NSTEP=100100
DECAY=0.95

MAX_ITER=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=2000 # interval for saving snapshot
SEED='1234'

###### Validation configs ######
SUPP_ID='[4]' #  # using the additionally loaded scan as support

echo ===================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
    for SUPERPIX_SCALE in "${ALL_SCALE[@]}"
    do
    PREFIX="train_${DATASET}_lbgroup${LABEL_SETS}_scale_${SUPERPIX_SCALE}_vfold${EVAL_FOLD}"
    echo $PREFIX
    LOGDIR="./examples/${CPT}_${SUPERPIX_SCALE}_${LABEL_SETS}"

    if [ ! -d $LOGDIR ]
    then
        mkdir $LOGDIR
    fi

    python3 training.py with \
    'modelname=dlfcn_res101' \
    'usealign=True' \
    'optim_type=sgd' \
    num_workers=$NWORKER \
    scan_per_load=-1 \
    label_sets=$LABEL_SETS \
    'use_wce=True' \
    exp_prefix=$PREFIX \
    'clsname=grid_proto' \
    n_steps=$NSTEP \
    exclude_cls_list=$EXCLU \
    eval_fold=$EVAL_FOLD \
    dataset=$DATASET \
    proto_grid_size=$PROTO_GRID \
    max_iters_per_load=$MAX_ITER \
    min_fg_data=1 seed=$SEED \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    superpix_scale=$SUPERPIX_SCALE \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR \
    support_idx=$SUPP_ID
    done
done

echo ===================================


