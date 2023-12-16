#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

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

    # "mpt": conv_mpt,


######### OkVQA-KO ####################
echo OKVQA-ko Start....


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader_okvqa \
        --model-path ${model_path} \
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

#######GQA###################################
echo GQA Start....
GQADIR="/data/MLP/cschoi/LLaVA/playground/data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${model_path} \
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

#ScienceQA
echo ================================================================================
echo ScienceQA Start....
python -m llava.eval.model_vqa_science \
    --model-path ${model_path} \
    --question-file /data/MLP/cschoi/LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /data/MLP/cschoi/LLaVA/playground/data/eval/scienceqa/images/test \
    --answers-file /data/MLP/cschoi/LLaVA/playground/data/eval/scienceqa/answers/${answers_file_name}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode ${conv_mode}
wait

python llava/eval/eval_science_qa.py \
    --base-dir /data/MLP/cschoi/LLaVA/playground/data/eval/scienceqa \
    --result-file /data/MLP/cschoi/LLaVA/playground/data/eval/scienceqa/answers/${answers_file_name}.jsonl \
    --output-file /data/MLP/cschoi/LLaVA/playground/data/eval/scienceqa/answers/${answers_file_name}_output.jsonl \
    --output-result /data/MLP/cschoi/LLaVA/playground/data/eval/scienceqa/answers/${answers_file_name}_result.json \
    --quantitive-file /data/MLP/cschoi/LLaVA/playground/data/eval/scienceqa/scores/${answers_file_name}.txt

wait
echo ScienceQA Done....
echo ================================================================================

# TextVQA
echo ================================================================================
echo TextVQA Start....
python -m llava.eval.model_vqa_loader \
    --model-path ${model_path} \
    --question-file /data/MLP/cschoi/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /data/MLP/cschoi/LLaVA/playground/data/eval/textvqa/train_images \
    --answers-file /data/MLP/cschoi/LLaVA/playground/data/eval/textvqa/answers/${answers_file_name}.jsonl \
    --temperature 0 \
    --conv-mode ${conv_mode}

wait

python -m llava.eval.eval_textvqa \
    --annotation-file /data/MLP/cschoi/LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /data/MLP/cschoi/LLaVA/playground/data/eval/textvqa/answers/${answers_file_name}.jsonl \
    --quantitive-file /data/MLP/cschoi/LLaVA/playground/data/eval/textvqa/scores/${answers_file_name}.txt
wait
echo TextVQA Done....
echo ================================================================================

# POPE
echo ================================================================================
echo POPE Start....
python -m llava.eval.model_vqa_loader \
    --model-path ${model_path} \
    --question-file /data/MLP/cschoi/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /data/MLP/cschoi/LLaVA/playground/data/eval/pope/val2014 \
    --answers-file /data/MLP/cschoi/LLaVA/playground/data/eval/pope/answers/${answers_file_name}.jsonl \
    --temperature 0 \
    --conv-mode ${conv_mode}
wait

python llava/eval/eval_pope.py \
    --annotation-dir /data/MLP/cschoi/LLaVA/playground/data/eval/pope/coco \
    --question-file /data/MLP/cschoi/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /data/MLP/cschoi/LLaVA/playground/data/eval/pope/answers/${answers_file_name}.jsonl \
    --quantitive-file /data/MLP/cschoi/LLaVA/playground/data/eval/pope/scores/${answers_file_name}.txt
wait
echo POPE Done....
echo ================================================================================

# MME
echo ================================================================================
echo MME Start....
python -m llava.eval.model_vqa_loader \
    --model-path ${model_path} \
    --question-file /data/MLP/cschoi/LLaVA/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /data/MLP/cschoi/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /data/MLP/cschoi/LLaVA/playground/data/eval/MME/answers/${answers_file_name}.jsonl \
    --temperature 0 \
    --conv-mode ${conv_mode}
wait

cd /data/MLP/cschoi/LLaVA/playground/data/eval/MME

python convert_answer_to_mme.py --experiment ${answers_file_name}

wait

cd /data/MLP/cschoi/LLaVA/playground/data/eval/MME/eval_tool

python calculation.py \
    --results_dir /data/MLP/cschoi/LLaVA/playground/data/eval/MME/eval_tool/answers/${answers_file_name} \
    --quantitive-file /data/MLP/cschoi/LLaVA/playground/data/eval/MME/scores/${answers_file_name}.txt

cd /data/MLP/cschoi/LLaVA
echo MME Done....
echo ================================================================================

echo merge scores txt.......

python ./scripts/merge_scores_to_one.py \
    --answer-file ${answers_file_name}.txt

echo the file is in /data/MLP/cschoi/LLaVA/playground/data/eval/quantitive_evaluation/${answers_file_name}.txt