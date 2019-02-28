#!/bin/bash


PATHDATASET='~/.datasets/'
NAMEDATASET='affectnet' #bu3dfe, ferblack, ck, affectnetdark, affectnet, ferp
PROJECT='../out'
PROJECTNAME='feratt_atentionresnet34_attgmm_adam_affectnetdark_dim64_preactresnet18x32_fold0_002'
PATHNAMEOUT='../out'
FILENAME='result.txt'
PATHMODEL='models'
NAMEMODEL='chk000048.pth.tar' #'model_best.pth.tar' #'chk000565.pth.tar'
MODEL=$PROJECT/$PROJECTNAME/$PATHMODEL/$NAMEMODEL  

python ../eval.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--pathnameout=$PATHNAMEOUT \
--filename=$FILENAME \
--model=$MODEL \


