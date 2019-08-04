#!/bin/bash


PATHDATASET='~/.datasets/'
NAMEDATASET='affectnet' #bu3dfe, ferblack, ck, affectnetdark, affectnet, ferp
PROJECT='../out'
PATHNAMEOUT='../out/attnet'
FILENAME='result.txt'
PATHMODEL='models'
NAMEMODEL='chk000080.pth.tar' #'model_best.pth.tar' #'chk000565.pth.tar'


PROJECTNAME='att_attgmmnet_ferattentiongmm_attloss_adam_affectnetdark_dim64_cvgg13x32_fold0_000'
MODEL=$PROJECT/$PROJECTNAME/$PATHMODEL/$NAMEMODEL  

python ../eval.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--pathnameout=$PATHNAMEOUT \
--filename=$FILENAME \
--model=$MODEL \


