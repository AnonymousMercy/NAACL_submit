gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

model_path=MLP-KTLim/X-LLaVA_O_A_BaseLLM_L
# conv_mode=vicuna_v1
answers_file_name=X-LLaVA_O_A_BaseLLM_L

output_file=/data/MLP/cschoi/LLaVA/playground/data/eval/okvqa_en/answers/$answers_file_name/merge_${answers_file_name}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
   cat /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa_en/answers/$answers_file_name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python scripts/convert_okvqa_en_for_score.py \
    --input-file ${output_file} \
    --output-file /data/MLP/cschoi/LLaVA/playground/data/eval/okvqa_en/scores/${answers_file_name}.txt

wait
echo BOKVQA-EN Done....