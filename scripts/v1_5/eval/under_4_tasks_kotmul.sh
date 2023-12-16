answers_file_name=X-LLaVA_O_A_BaseLLM_L


cd /data/MLP/cschoi/LLaVA
echo merge scores txt.......

python ./scripts/merge_scores_to_one_internal.py \
    --answer-file ${answers_file_name}.txt

echo the file is in /data/MLP/cschoi/LLaVA/playground/data/eval/quantitive_evaluation/${answers_file_name}.txt


