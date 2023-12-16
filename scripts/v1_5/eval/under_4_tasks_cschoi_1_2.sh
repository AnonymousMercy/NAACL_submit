#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}


# model_path=MLP-KTLim/X-LLaVA_VQA_O_BaseLLM_L
# conv_mode=llava_llama_2
# answers_file_name=X-LLaVA_VQA_O_BaseLLM_L

# model_path=liuhaotian/llava-v1.5-13b

model_path=MLP-KTLim/X-LLaVA_VQA_O_BaseLLM_L
conv_mode=llava_llama_2
answers_file_name=X-LLaVA_VQA_O_BaseLLM_L


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


######### OkVQA-KO ####################
echo OKVQA-ko Start....


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader_okvqa \
        --model-path ${model_path} \
        --model-base ${model_base} \
        --question-file /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/okvqa_test2015_modify.jsonl \
        --image-folder /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/img \
        --answers-file /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/answers/$answers_file_name/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode ${conv_mode} &
done

wait

output_file=/data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/answers/$answers_file_name/merge_${answers_file_name}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
   cat /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/answers/$answers_file_name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python scripts/convert_okvqa_for_score.py \
    --input-file ${output_file} \
    --output-file /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/scores/${answers_file_name}.txt

wait
echo OKVQA-ko Done....

#######GQA###################################
echo GQA Start....
GQADIR="/data/MLP/cschoi/LLaVA/playground/data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${model_path} \
        --model-base ${model_base} \
        --question-file /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl \
        --image-folder /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/data/images \
        --answers-file /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/answers/${answers_file_name}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $conv_mode &
done

wait

output_file=/data/MLP/cschoi/LLaVA/playground/data/eval/gqa/answers/${answers_file_name}/merge_${answers_file_name}.jsonl

# # Clear out the output file if it exists.
> "$output_file"

# # Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/answers/${answers_file_name}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src ${output_file} --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced --quantitive-file /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/scores/${answers_file_name}.txt


cd /data/MLP/cschoi/LLaVA

echo GQA Done....

##########################################################################################


cd /data/MLP/cschoi/LLaVA
echo merge scores txt.......

python ./scripts/merge_scores_to_one_internal.py \
    --answer-file ${answers_file_name}.txt

echo the file is in /data/MLP/cschoi/LLaVA/playground/data/eval/quantitive_evaluation/${answers_file_name}.txt




#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# ######### OkVQA-KO ####################
# echo OKVQA-ko Start....

# # model_path=liuhaotian/llava-v1.5-13b
# model_path=MLP-KTLim/X-LLaVA_S2_VQA_BaseLLM_T_L
# conv_mode=llava_llama_2
# answers_file_name=X-LLaVA_S2_VQA_BaseLLM_T_L

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader_okvqa \
#         --model-path ${model_path} \
#         --question-file /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/okvqa_test2015_modify.jsonl \
#         --image-folder /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/img \
#         --answers-file /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/answers/$answers_file_name/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode ${conv_mode} &
# done

# wait

# output_file=/data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/answers/$answers_file_name/merge_${answers_file_name}.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#    cat /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/answers/$answers_file_name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done


# python scripts/convert_okvqa_for_score.py \
#     --input-file ${output_file} \
#     --output-file /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/scores/${answers_file_name}.txt

# wait
# echo OKVQA-ko Done....

# #######GQA###################################
# echo GQA Start....
# GQADIR="/data/MLP/cschoi/LLaVA/playground/data/eval/gqa/data"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path ${model_path} \
#         --question-file /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl \
#         --image-folder /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/data/images \
#         --answers-file /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/answers/${answers_file_name}/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode $conv_mode &
# done

# wait

# output_file=/data/MLP/cschoi/LLaVA/playground/data/eval/gqa/answers/${answers_file_name}/merge_${answers_file_name}.jsonl

# # # Clear out the output file if it exists.
# > "$output_file"

# # # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/answers/${answers_file_name}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# python scripts/convert_gqa_for_eval.py --src ${output_file} --dst $GQADIR/testdev_balanced_predictions.json

# cd $GQADIR
# python eval/eval.py --tier testdev_balanced --quantitive-file /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/scores/${answers_file_name}.txt


# cd /data/MLP/cschoi/LLaVA

# echo GQA Done....

# ##########################################################################################


# cd /data/MLP/cschoi/LLaVA
# echo merge scores txt.......

# python ./scripts/merge_scores_to_one_internal.py \
#     --answer-file ${answers_file_name}.txt

# echo the file is in /data/MLP/cschoi/LLaVA/playground/data/eval/quantitive_evaluation/${answers_file_name}.txt


# #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# ######### OkVQA-KO ####################
# echo OKVQA-ko Start....

# # model_path=liuhaotian/llava-v1.5-13b
# model_path=MLP-KTLim/X-LLaVA_S2_VQA_BaseLLM
# conv_mode=llava_llama_2
# answers_file_name=X-LLaVA_S2_VQA_BaseLLM

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader_okvqa \
#         --model-path ${model_path} \
#         --question-file /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/okvqa_test2015_modify.jsonl \
#         --image-folder /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/img \
#         --answers-file /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/answers/$answers_file_name/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode ${conv_mode} &
# done

# wait

# output_file=/data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/answers/$answers_file_name/merge_${answers_file_name}.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#    cat /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/answers/$answers_file_name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done


# python scripts/convert_okvqa_for_score.py \
#     --input-file ${output_file} \
#     --output-file /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/scores/${answers_file_name}.txt

# wait
# echo OKVQA-ko Done....

# #######GQA###################################
# echo GQA Start....
# GQADIR="/data/MLP/cschoi/LLaVA/playground/data/eval/gqa/data"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path ${model_path} \
#         --question-file /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl \
#         --image-folder /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/data/images \
#         --answers-file /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/answers/${answers_file_name}/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode $conv_mode &
# done

# wait

# output_file=/data/MLP/cschoi/LLaVA/playground/data/eval/gqa/answers/${answers_file_name}/merge_${answers_file_name}.jsonl

# # # Clear out the output file if it exists.
# > "$output_file"

# # # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/answers/${answers_file_name}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# python scripts/convert_gqa_for_eval.py --src ${output_file} --dst $GQADIR/testdev_balanced_predictions.json

# cd $GQADIR
# python eval/eval.py --tier testdev_balanced --quantitive-file /data/MLP/cschoi/LLaVA/playground/data/eval/gqa/scores/${answers_file_name}.txt


# cd /data/MLP/cschoi/LLaVA

# echo GQA Done....

# ##########################################################################################


# cd /data/MLP/cschoi/LLaVA
# echo merge scores txt.......

# python ./scripts/merge_scores_to_one_internal.py \
#     --answer-file ${answers_file_name}.txt

# echo the file is in /data/MLP/cschoi/LLaVA/playground/data/eval/quantitive_evaluation/${answers_file_name}.txt