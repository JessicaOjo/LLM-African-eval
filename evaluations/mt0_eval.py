import glob
import json
import argparse
import pandas as pd
from sklearn.metrics import f1_score

from evaluations import utils


def qa_eval(predicted_files, metrics):
    for file in predicted_files:
        df = pd.read_csv(file, sep='\t')
        df['translated_answer'] = df['answer_pivot'].apply(lambda x: x.split(': ')[-1].strip("['").rstrip("']}"))
        df['mt0'] = df['mt0'].str.split("<pad>").str[-1].str.split("</s>").str[0].str.strip()

        df.fillna('', inplace=True)

        df = df[~(df['translated_answer'] == '')]
        df[['mt0', 'translated_answer']] = df[['mt0', 'translated_answer']].applymap(utils.normalize_text)

        language = file.split('/')[-1].split('.')[-2]

        lang_metric = utils.calculate_qa_metrics(df, 'mt0')
        metrics[language] = lang_metric
    return metrics


def mt_eval(predicted_files, metrics):
    for file in predicted_files:
        df = pd.read_csv(file, sep='\t')
        df['mt0'] = df['mt0'].str.split("<pad>").str[-1].str.split("</s>").str[0].str.strip()

        lang_full = file.split('/')[-1].split('.')[0]
        lang = lang_full.split('-')[1]
        if lang == 'eng':
            lang = 'eng_Latn'
        elif lang == 'fra':
            lang = 'fra_Latn'
        elif lang == 'deu':
            lang = 'deu_Latn'

        df[[lang, 'mt0']] = df[[lang, 'mt0']].applymap(utils.normalize_text)

        lang_metric = utils.calculate_mt_metrics(df, 'mt0', lang)
        metrics[lang_full] = lang_metric
    return metrics


def senti_eval(predicted_files, metrics):
    for file in predicted_files:
        df = pd.read_csv(file, sep='\t')
        df['mt0'] = df['mt0'].str.split("<pad>").str[-1].str.split("</s>").str[0].str.strip()

        df = utils.filter_mt0_labels(df)

        lang = file.split('/')[-1].replace('.tsv', '')
        language = utils.language_abv(lang)
        metrics[language] = round((f1_score(df['label'], df['mt0'], average='weighted') * 100), 2)
    return metrics


def news_eval(predicted_files, metrics):
    for file in predicted_files:
        df = pd.read_csv(file, sep='\t')
        df['mt0'] = df['mt0'].str.split("<pad>").str[-1].str.split("</s>").str[0].str.strip()

        df = df.fillna("unknown")
        df['mt0'] = df['mt0'].apply(lambda x: x.split('/')[0])
        df[['category', 'mt0']] = df[['category', 'mt0']].applymap(utils.normalize_text)
        df['mt0'] = df['mt0'].apply(utils.verbalizer)

        # if it contains more than one label
        df['mt0'] = df['mt0'].apply(lambda x: "unknown" if x.count(' ') >= 1 else x)

        # assign random labels to unknowns
        df['mt0'] = df.apply(utils.assign_label, axis=1, row_name='mt0')

        f1 = round((f1_score(df['category'], df['mt0'], average='weighted') * 100), 2)

        language = file.split('/')[-1].replace('.tsv', '')
        metrics[language] = f1
    return metrics


def ner_eval(predicted_files, metrics):
    for file in predicted_files:
        df = pd.read_csv(file, sep='\t', header=0)
        df['mt0'] = df['mt0'].str.split("<pad>").str[-1].str.split("</s>").str[0].str.strip()

        df.fillna('none', inplace=True)

        df['target'] = df['target'].apply(utils.format_ner_text, target=True)
        df['mt0'] = df['mt0'].apply(utils.format_ner_text, target=False)
        df = df[~(df.target == '')]

        f1 = utils.calculate_ner_metrics(df, 'mt0')
        language = file.split('/')[-1].split('.')[0]
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
    parser.add_argument('--task_function', type=str, default=news_eval)
    parser.add_argument('--prediction_directory', type=str, default='../predictions/results_mt0/news_topic')
    parser.add_argument('--output_directory', type=str, default='../results/mt0/')

    args = parser.parse_known_args()
    main(args)
