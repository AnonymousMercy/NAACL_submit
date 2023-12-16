import os
import argparse

tasks = ['okvqa', 'gqa']
base_dir = "/data/MLP/cschoi/LLaVA/playground/data/eval"
dst_dir = '/data/MLP/cschoi/LLaVA/playground/data/eval/quantitive_evaluation'

parse = argparse.ArgumentParser()
parse.add_argument('--answer-file', type=str)
args = parse.parse_args()

with open(os.path.join(dst_dir, args.answer_file), 'w') as file:
    file.write(args.answer_file.split('.txt')[0] + '\n')

for task in tasks:
    with open(os.path.join(dst_dir, args.answer_file), 'a') as file:
        with open(os.path.join(base_dir, task, 'scores', args.answer_file), 'r') as tgt:
            file.write(tgt.read())
            file.write('\n\n')