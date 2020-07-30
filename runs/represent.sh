#!/bin/bash

PATHDATASET='~/.datasets/'
NAMEDATASET='ferp' #bu3dfe, ferblack, ck, celeba, affectnetdark, affectnet, ferp
PROJECT='../out'
PROJECTNAME='feratt_atentionresnet34_attgmm_adam_affectnetdark_dim64_preactresnet18x32_fold0_000'
PATHNAMEOUT='../out'
FILENAME='result.txt'
PATHMODEL='models'
NAMEMODEL='model_best.pth.tar' #'model_best.pth.tar' #'chk000565.pth.tar'
MODEL=$PROJECT/$PROJECTNAME/$PATHMODEL/$NAMEMODEL

python ../represent.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--pathnameout=$PATHNAMEOUT \
--filename=$FILENAME \
--model=$MODEL \
