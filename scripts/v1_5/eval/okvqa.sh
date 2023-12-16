# #!/bin/bash

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CKPT="llava-v1.5-13b"
# SPLIT="okvqa_test2015"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader_okvqa \
#         --model-path liuhaotian/llava-v1.5-13b \
#         --question-file ./playground/data/eval/okvqa/$SPLIT.jsonl \
#         --image-folder ./playground/data/eval/okvqa/img \
#         --answers-file ./playground/data/eval/okvqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# # conv-mode llava_llama_2 도 있음

# wait

# output_file=./playground/data/eval/okvqa/answers/$SPLIT/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#    cat ./playground/data/eval/okvqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# # python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT















gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

model_path=liuhaotian/llava-v1.5-13b
conv_mode=vicuna_v1
answers_file_name=llava-v1.5-13b

######### OkVQA-EN ####################
echo OKVQA-en Start....


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader_okvqa \
        --model-path ${model_path} \
        --question-file /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa_en/okvqa_test2015_en.jsonl \
        --image-folder /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa_en/img \
        --answers-file /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa_en/answers_en/$answers_file_name/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode ${conv_mode} &
done

wait

output_file=/data/MLP/cschoi/LLaVA/playground/data/eval/okvqa_en/answers_en/$answers_file_name/merge_${answers_file_name}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
   cat /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa_en/answers_en/$answers_file_name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python scripts/convert_okvqa_en_for_score.py \
    --input-file ${output_file} \
    --output-file /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa_en/scores/${answers_file_name}.txt

wait
echo OKVQA-EN Done....