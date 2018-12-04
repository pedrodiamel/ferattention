#!/bin/bash


# parameters
DATABACK='~/.datasets/coco'
DATA='~/.datasets'
NAMEDATASET='ferblack' #bu3dfe, ferblack
PROJECT='../out'
EPOCHS=500
BATCHSIZE=240
LEARNING_RATE=0.0001
MOMENTUM=0.5
PRINT_FREQ=75
WORKERS=80
RESUME='chk000000xx.pth.tar'
GPU=0
ARCH='atentionresnet34'
LOSS='attgmm'
OPT='adam'
SCHEDULER='step'
NUMCLASS=8
NUMCHANNELS=3
DIM=64
SNAPSHOT=10
IMAGESIZE=128
KFOLD=0
NACTOR=10
EXP_NAME='fer_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_dim'$DIM'_preactresnet18x32_fold'$KFOLD'_002'


rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT    
mkdir $PROJECT/$EXP_NAME  


python ../train.py \
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
--arch=$ARCH \
--parallel \
--finetuning \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

