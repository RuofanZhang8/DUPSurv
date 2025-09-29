#!/bin/bash  

gpuid=$1
config=$2

task='TASK_NAME'
target_col='TARGET_COLUMN'
split_names='train,test,val'
data_source='PATH_TO_FEATURES'
result_path='PATH_TO_RESULTS'

for k in 0 1 2 3 4; do
    split_dir="PATH_TO_SPLITS/k=${k}"
    feat_name='FEATURE_NAME'
    tags="FEATURE_NAME_survival_k=${k}_TAG"
    
    bash "./scripts/survival/${config}.sh" \
        $gpuid \
        $task \
        $target_col \
        $split_dir \
        $split_names \
        $data_source \
        $feat_name \
        $tags 
done
