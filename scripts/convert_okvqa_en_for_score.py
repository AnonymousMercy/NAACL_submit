import json
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys

def load_data(filename):
    with open(filename, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def calculate_metrics(data):
    y_true = [item['label'].lower() for item in data]
    y_pred = [item['label'].lower() if item['label'].lower() in item['text'].lower() else item['text'].lower() for item in data]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    return accuracy, precision, recall, f1

def print_and_save_metrics(metrics, output_file):
    with open(output_file, 'w') as file:
        file.write('=' * 40 + '\n')
        file.write('OK-VQA-EN\n')

    acc, precision, recall, f1 = metrics
    print('output_file_path: ', output_file)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    with open(output_file, 'a') as file:
        file.write('Acc\tPrecision\tRecall\tF1\n')
        file.write('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t'.format(acc, precision, recall, f1))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate metrics from a JSONL file.')
    parser.add_argument('--input-file', type=str, help='Path to the JSONL input file')
    parser.add_argument('--output-file', type=str, help='Path to save the output metrics')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    data = load_data(args.input_file)
    metrics = calculate_metrics(data)
    print_and_save_metrics(metrics, args.output_file)
