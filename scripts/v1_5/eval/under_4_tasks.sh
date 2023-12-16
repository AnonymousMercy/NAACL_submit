#!/bin/bash
model_path=MLP-KTLim/X-LLaVA_O_A_BaseLLM_L
conv_mode=llava_llama_2
answers_file_name=X-LLaVA_O_A_BaseLLM_L

# ScienceQA
python -m llava.eval.model_vqa_science \
    --model-path ${model_path} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${answers_file_name}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode ${conv_mode}

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${answers_file_name}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${answers_file_name}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${answers_file_name}_result.json


# # TextVQA
# python -m llava.eval.model_vqa_loader \
#     --model-path ${model_path} \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder ./playground/data/eval/textvqa/train_images \
#     --answers-file ./playground/data/eval/textvqa/answers/${answers_file_name}.jsonl \
#     --temperature 0 \
#     --conv-mode ${conv_mode}

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/${answers_file_name}.jsonl


# POPE
python -m llava.eval.model_vqa_loader \
    --model-path ${model_path} \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/${answers_file_name}.jsonl \
    --temperature 0 \
    --conv-mode ${conv_mode}

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/${answers_file_name}.jsonl


# # MME
# python -m llava.eval.model_vqa_loader \
#     --model-path ${model_path} \
#     --question-file ./playground/data/eval/MME/llava_mme.jsonl \
#     --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
#     --answers-file ./playground/data/eval/MME/answers/${answers_file_name}.jsonl \
#     --temperature 0 \
#     --conv-mode ${conv_mode}

# cd ./playground/data/eval/MME

# python convert_answer_to_mme.py --experiment ${answers_file_name}

# cd eval_tool

# python calculation.py --results_dir answers/${answers_file_name}

# cd ../../../../../