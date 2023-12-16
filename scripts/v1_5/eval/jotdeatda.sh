model_path=tabtoyou/X-LLaVA_O_A_BaseLLM_L
conv_mode=llava_llama_2
answers_file_name=X-LLaVA_O_A_BaseLLM_L


#################### ok-VQA ####################

echo ok-vqa

output_file=/data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/answers/$answers_file_name/merge_${answers_file_name}.jsonl

python scripts/convert_okvqa_for_score.py \
    --input-file ${output_file} \
    --output-file /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa/scores/jotjot/${answers_file_name}.txt

wait
echo OKVQA-ko Done....


#################### Ko-VQA ####################
echo Ko-VQA Start....

output_file=/data/MLP/cschoi/LLaVA/playground/data/eval/kovqa/answers/$answers_file_name/merge_${answers_file_name}.jsonl

python scripts/convert_kovqa_for_score.py \
    --input-file ${output_file} \
    --output-file /data/MLP/cschoi/LLaVA/playground/data/eval/kovqa/scores/jotjot/${answers_file_name}.txt

wait
echo Ko-VQA Done....

#################### Living-VQA ####################
echo Living-VQA Start....

output_file=/data/MLP/cschoi/LLaVA/playground/data/eval/livingvqa/answers/$answers_file_name/merge_${answers_file_name}.jsonl

python scripts/convert_livingvqa_for_score.py \
    --input-file ${output_file} \
    --output-file /data/MLP/cschoi/LLaVA/playground/data/eval/livingvqa/scores/jotjot/${answers_file_name}.txt

wait
echo Living-VQA Done....