#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

# 필요하면
# model_base=kfkas/Llama-2-ko-7b-Chat
# model_path=liuhaotian/llava-v1.5-13b
model_path=MLP-KTLim/X-LLaVA_Only_China_A_BaseLLM_L
conv_mode=llava_llama_2
answers_file_name=X-LLaVA_Only_China_A_BaseLLM_L
# model_base=kfkas/Llama-2-ko-7b-Chat



# 프롬프트 템플릿 종류
    # "default": conv_vicuna_v0,
    # "v0": conv_vicuna_v0
    # "v1": conv_vicuna_v1,
    # "vicuna_v1": conv_vicuna_v1,
    # "llama_2": conv_llama_2,

    # "plain": conv_llava_plain,
    # "v0_plain": conv_llava_plain,
    # "llava_v0": conv_llava_v0,
    # "v0_mmtag": conv_llava_v0_mmtag,
    # "llava_v1": conv_llava_v1,
    # "v1_mmtag": conv_llava_v1_mmtag,
    # "llava_llama_2": conv_llava_llama_2,
    # mistral = synatra
    # "mpt": conv_mpt,


######### zh-vqa ####################
echo zh-vqa, fm_iqa Start....


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader_okvqa \
        --model-path ${model_path} \
        --question-file /data/MLP/cschoi/LLaVA/playground/data/eval/cn_vqa/fm_iqa_eval.jsonl \
        --image-folder /data/MLP/cschoi/LLaVA/playground/data/eval/cn_vqa/img \
        --answers-file /data/MLP/cschoi/LLaVA/playground/data/eval/cn_vqa/answers/$answers_file_name/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode ${conv_mode} &
done

wait

output_file=/data/MLP/cschoi/LLaVA/playground/data/eval/cn_vqa/answers/$answers_file_name/merge_${answers_file_name}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
   cat /data/MLP/cschoi/LLaVA/playground/data/eval/cn_vqa/answers/$answers_file_name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python scripts/convert_okvqa_for_score.py \
    --input-file ${output_file} \
    --output-file /data/MLP/cschoi/LLaVA/playground/data/eval/cn_vqa/scores/${answers_file_name}.txt

wait
echo zh-vqa, fm_iqa Done....