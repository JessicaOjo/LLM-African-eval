import glob
import json
import evaluate
import argparse
import pandas as pd
from evaluations import utils
from sklearn.metrics import f1_score


def senti_eval(predicted_files, metric):
    for file in predicted_files:
        df = pd.read_csv(file, sep='\t', header=0)
        df.drop('Unnamed: 0', axis=1, inplace=True)
        df.rename({'gpt-turbo': 'gpt-4'}, axis=1, inplace=True)
        lang = file.split('/')[-1].split('.')[0]
        labels = ["negative", "positive", "neutral"]
        for label in labels:
            df['gpt-4'][(df['gpt-4'] != 'positive') & (df['gpt-4'] != 'negative')
                        & (df['gpt-4'] != 'neutral') & (df['label'] != label)] = label

        metric[lang] = round((f1_score(df['label'], df['gpt-4'], average='weighted')*100), 2)
    return metric


def ner_eval(predicted_files, metric):
    for file in predicted_files:
        df = pd.read_csv(file, sep='\t', header=0)
        df['target'] = df['target'].apply(utils.format_ner_text, target=True)
        df['gpt-4'] = df['gpt-4'].apply(utils.format_ner_text, target=False)
        df = df[~(df.target == '')]

        language = file.split('/')[-1].split('.')[0]
        metric[language] = utils.calculate_ner_metrics(df, 'gpt-4')
    return metric


def mt_eval(predicted_files, metrics):
    for file in predicted_files:
        df = pd.read_csv(file, sep='\t', header=0)

        lang_full = file.split('/')[-1].split('.')[0]
        lang = file.split('/')[-1].split('.')[0].split('-')[1]

        if lang == 'eng':
            language = 'eng_Latn'
        elif lang == 'fra':
            language = 'fra_Latn'
        elif lang == 'deu':
            language = 'deu_Latn'
        else:
            language = lang_full.split('-')[1]

        df[[language, 'gpt-4']] = df[[language, 'gpt-4']].applymap(utils.normalize_text)

        lang_metric = utils.calculate_mt_metrics(df, 'gpt-4', language)
        metrics[lang_full] = lang_metric
    return metrics


def qa_eval(predicted_files, metrics):
    for file in predicted_files:
        df = pd.read_csv(file, sep='\t', header=0)
        df.rename({'gpt-turbo': 'gpt-4'}, axis=1, inplace=True)
        df['translated_answer'] = df['answer_pivot'].apply(lambda x: x.split(': ')[-1].strip("['").rstrip("']}"))
        df[['gpt-4', 'translated_answer']] = df[['gpt-4', 'translated_answer']].applymap(utils.normalize_text)
        df = df[~(df['translated_answer'] == '')]

        language = file.split('/')[-1].split('.')[-2]

        lang_metric = utils.calculate_qa_metrics(df, 'gpt-4')
        metrics[language] = lang_metric
    return metrics


def news_eval(predicted_files, metrics):
    for file in predicted_files:
        df = pd.read_csv(file, sep='\t', header=0)
        df.drop('Unnamed: 0', axis=1, inplace=True)
        language = file.split('/')[-1].replace('.tsv', '')
        df = utils.gpt4_match_news_columns(df)
        labels = ["business", "entertainment", "health", "politics", "religion", "sports", "technology"]
        for label in labels:
            df['gpt-4-clean'][(df['gpt-4-clean'] == 'unknown') & (df['category'] != label)] = label

        f1 = round((f1_score(df['category'], df['gpt-4-clean'], average='weighted') * 100), 2)
        metrics[language] = f1
    return metrics


def main(args):
    args, unknown = args
    task_function = args.task_function
    prediction_dir = args.prediction_directory
    output_dir = args.output_directory

    prediction_files = glob.glob(f'{prediction_dir}/*tsv', recursive=True)
    metrics = {}

    results = task_function(prediction_files, metrics)
    utils.create_dir(output_dir)
    task_name = prediction_dir.split('/')[-1]
    with open(f'{output_dir}/{task_name}', 'w') as outfile:
        json.dump(results, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_function', default=senti_eval)
    parser.add_argument('--prediction_directory', type=str, default='../predictions/results_gpt4/sentiment')
    parser.add_argument('--output_directory', type=str, default='../results/gpt4/')

    args = parser.parse_known_args()
    main(args)
