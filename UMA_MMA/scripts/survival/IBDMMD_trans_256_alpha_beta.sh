#!/bin/bash
gpuid=$1
task=$2
target_col=$3
split_dir=$4
split_names=$5
dataroots=("$@")
data_source=$6
feat_name=$7
tags=$8
diff_step=<DIFF_NUM>
resultmy_path=<${10}>
alpha=0.1
beta=0.1

feat=<FEATURE_NAME>
input_dim=<INPUT_DIM>
mag='20x'
patch_size=256
bag_size='-1'
batch_size=<BATCH_SIZE>
out_size=16
out_type='allcat'
model_tuple='PANTHER,default'
lin_emb_model="coattn_IBDMMD_transformer_256_net_indiv_alpha_${alpha}_beta_${beta}"   #LinearEmb IndivMLPEmb_Indiv coattn_mot     IndivMLPEmb_Shared
echo "Using linear embedding model: ${lin_emb_model}"
max_epoch=50
lr=0.0001
wd=0.00001
lr_scheduler='cosine'  #constant cosine
opt='adamW'
grad_accum=1
loss_fn='cox'
n_label_bin=4
em_step=1
load_proto=1
es_flag=0
tau=1.0
eps=1
n_fc_layer=0
proto_num_samples='1.0e+05'
save_dir_root=results

IFS=',' read -r model config_suffix <<< "${model_tuple}"
model_config=${model}_${config_suffix}

exp_code=${task}::${model_config}::${feat_name}
save_dir=${save_dir_root}/${exp_code}

th=0.00005
if awk "BEGIN {exit !($lr <= $th)}"; then
  warmup=0
  curr_lr_scheduler='constant'
else
  curr_lr_scheduler=$lr_scheduler
  warmup=1
fi


# Actual command
cmd="CUDA_VISIBLE_DEVICES=$gpuid python -m training.main_survival_IBD \\
--tags ${tags} \\
--data_source ${data_source} \\
--results_dir ${save_dir} \\
--split_dir ${split_dir} \\
--split_names ${split_names} \\
--task ${task} \\
--target_col ${target_col} \\
--model_type ${model} \\
--model_config ${model}_default \\
--n_fc_layers ${n_fc_layer} \\
--in_dim ${input_dim} \\
--opt ${opt} \\
--lr ${lr} \\
--lr_scheduler ${curr_lr_scheduler} \\
--accum_steps ${grad_accum} \\
--wd ${wd} \\
--warmup_epochs ${warmup} \\
--max_epochs ${max_epoch} \\
--train_bag_size ${bag_size} \\
--batch_size ${batch_size} \\
--in_dropout 0 \\
--seed 1 \\
--num_workers 4 \\
--em_iter ${em_step} \\
--tau ${tau} \\
--n_proto ${out_size} \\
--out_type ${out_type} \\
--loss_fn ${loss_fn} \\
--nll_alpha ${alpha} \\
--n_label_bins ${n_label_bin} \\
--early_stopping ${es_flag} \\
--emb_model_type ${lin_emb_model} \\
--ot_eps ${eps} \\
--fix_proto \\
--diff_step ${diff_step} \\
--resultmy_path ${resultmy_path} \\
--dump_path ${resultmy_path} \\
--only_MILemb \\
"


# Specify prototype path if load_proto is True
if [[ $load_proto -eq 1 ]]; then
  cmd="$cmd --load_proto \\
  --proto_path \"${split_dir}/prototypes/prototypes_c${out_size}_${feat_name}_faiss_num_${proto_num_samples}.pkl\" \\
  "
fi

eval "$cmd"
