#!/bin/bash

# parameters
DATABACK='~/.datasets/coco'
DATA='~/.datasets'
NAMEDATASET='affectnetdark' #affectnetdark, bu3dfedark, ckdark, jaffedark
PROJECT='../out/mitosis'
EPOCHS=20
BATCHSIZE=640 #64, 128, 192, 240, 256, 456
LEARNING_RATE=0.0001
MOMENTUM=0.5
PRINT_FREQ=75
WORKERS=20
RESUME='chk000390.pth.tar' #chk000000, model_best
GPU=0
NAMEMETHOD='mitosisattgmmnet' # mitosisattgmmnet
ARCH='ferattentiongmm' #ferattention, ferattentiongmm, ferattentionstn
LOSS='attloss'
OPT='adam'
SCHEDULER='fixed'
NUMCLASS=8 #6, 7, 8
NUMCHANNELS=3
DIM=64
SNAPSHOT=10
IMAGESIZE=64
KFOLD=0
NACTOR=10
EXP_NAME='mitosis_att_'$NAMEMETHOD'_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_dim'$DIM'_resnet18x32_fold'$KFOLD'_mixup_retrain_000'


# rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
# rm -rf $PROJECT/$EXP_NAME/
# mkdir -p $PROJECT
# mkdir -p $PROJECT/$EXP_NAME


CUDA_VISIBLE_DEVICES=0,1,2,3 python ../train_mitosis.py \
$DATA \
--databack=$DATABACK \
--name-dataset=$NAMEDATASET \
--project=$PROJECT \
--name=$EXP_NAME \
--epochs=$EPOCHS \
--kfold=$KFOLD \
--nactor=$NACTOR \
--batch-size=$BATCHSIZE \
--learning-rate=$LEARNING_RATE \
--momentum=$MOMENTUM \
--image-size=$IMAGESIZE \
--channels=$NUMCHANNELS \
--dim=$DIM \
--num-classes=$NUMCLASS \
--print-freq=$PRINT_FREQ \
--snapshot=$SNAPSHOT \
--workers=$WORKERS \
--resume=$RESUME \
--gpu=$GPU \
--loss=$LOSS \
--opt=$OPT \
--scheduler=$SCHEDULER \
--name-method=$NAMEMETHOD \
--arch=$ARCH \
--parallel \
--finetuning \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \
