#!/bin/bash


# parameters
DATABACK='~/.datasets/coco'
DATA='~/.datasets'
NAMEDATASET='bu3dfe' #ferblack
PROJECT='../out'
EPOCHS=65
BATCHSIZE=250
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
SNAPSHOT=5
IMAGESIZE=128
EXP_NAME='attentionrec_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_dim'$DIM'_preactresnet18x32_fold01_000'

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

