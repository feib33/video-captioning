#!/usr/bin/env bash

ctx_mode=video_sub  # [video, sub, video_sub]
vid_feat_type=resnet_i3d  # [resnet, i3d, resnet_i3d]
model_names=('2to1stream_self' '2to1stream_cross' '2streams_self' '2streams_cross' '2streams_dec')

max_v_len=20
max_sub_len=30
max_cap_len=20
train_path=data/tvc_train_release.jsonl
eval_train_path=data/tvc_val_release.jsonl
sub_meta_path=data/tvqa_preprocessed_subtitles.jsonl
word2idx_path=cache/tvc_word2idx.json
feature_root=data/tvc_feature_release
results_root=baselines/multimodal_transformer/results

if [[ ${vid_feat_type} == "i3d" ]]; then
    echo "Using I3D feature with shape 1024"
    vid_feat_path=${feature_root}/video_feature/tvr_i3d_rgb600_avg_cl-1.5.h5
    vid_feat_size=1024
elif [[ ${vid_feat_type} == "resnet" ]]; then
    echo "Using ResNet feature with shape 2048"
    vid_feat_path=${feature_root}/video_feature/tvr_resnet152_rgb_max_cl-1.5.h5
    vid_feat_size=2048
elif [[ ${vid_feat_type} == "resnet_i3d" ]]; then
    echo "Using concatenated ResNet and I3D feature with shape 2048+1024"
    vid_feat_path=${feature_root}/video_feature/tvr_resnet152_rgb_max_i3d_rgb600_avg_cat_cl-1.5.h5
    vid_feat_size=3072
fi

if [[ ${ctx_mode} != *"video"* ]] && [[ ${ctx_mode} != "video" ]]; then
    echo "Not using video, use fake video as inputs"
    vid_feat_size=2  # since we still have a fake video inputs
    max_v_len=2
fi

if [[ ${ctx_mode} != *"sub"* ]] && [[ ${ctx_mode} != "sub" ]]; then
    echo "Not using sub, use fake sub as inputs"
    max_sub_len=2
fi

split_name=val

eval_path=data/tvc_${split_name}_release.jsonl
extra_args=()
if [[ ${split_name} == val ]]; then
    reference_path=data/tvc_${split_name}_release.jsonl
    extra_args+=(-reference_path)
    extra_args+=(${reference_path})
fi

for name in "${model_names[@]}"
do
    echo "Training time for model ${name}"
    python baselines/multimodal_transformer/train.py \
    -ctx_mode ${ctx_mode} \
    -train_path ${train_path} \
    -eval_path ${eval_train_path} \
    -reference_path ${eval_train_path} \
    -sub_meta_path ${sub_meta_path} \
    -word2idx_path ${word2idx_path} \
    -max_v_len ${max_v_len} \
    -max_sub_len ${max_sub_len} \
    -max_cap_len ${max_cap_len} \
    -vid_feat_path ${vid_feat_path} \
    -vid_feat_size ${vid_feat_size} \
    -res_root_dir ${results_root} \
    -model_name ${name} \
    -no_norm_vfeat
    echo "Validating time for model ${name}"
    python baselines/multimodal_transformer/translate.py \
    -res_dir=${results_root}/${name}-default \
    -eval_split_name=${split_name} \
    -eval_path=${eval_path} \
    -model_name ${name} \
    ${extra_args[@]} \
    ${@:3}
done


