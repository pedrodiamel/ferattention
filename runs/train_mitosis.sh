#!/bin/bash

# parameters
DATABACK='~/.datasets/coco'
DATA='~/.datasets'
NAMEDATASET='affectnetdark' #affectnetdark, bu3dfedark, ckdark, jaffedark
PROJECT='../out'
EPOCHS=10
BATCHSIZE=512 #64, 128, 192, 240, 256
LEARNING_RATE=0.0001
MOMENTUM=0.5
PRINT_FREQ=100
WORKERS=30
RESUME='model_best.pth.tar' #chk000010, model_best
GPU=0
NAMEMETHOD='mitosisattgmmnet' # mitosisattgmmnet
ARCH='ferattentiongmm' #ferattention, ferattentiongmm, ferattentionstn
LOSS='attloss'
OPT='adam'
SCHEDULER='step'
NUMCLASS=8 #6, 7, 8
NUMCHANNELS=3
DIM=64
SNAPSHOT=10
IMAGESIZE=64
KFOLD=0
NACTOR=10
EXP_NAME='mitosis_'$NAMEMETHOD'_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_dim'$DIM'_preactresnet18x32_fold'$KFOLD'_001'


rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT    
mkdir $PROJECT/$EXP_NAME  


python ../train_mitosis.py \
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

