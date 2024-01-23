import glob
import json
import argparse
import pandas as pd
from sklearn.metrics import f1_score

from evaluations import utils


def senti_eval(prediction_files, metric):
    for file in prediction_files:
        df = pd.read_csv(file, sep='\t')
        df['llama_split'] = df['llama'].str.lower()
        df['llama_split'] = df['llama_split'].str.replace('\n', ' ', regex=True)
        df['llama_split'] = df['llama_split'].str.replace(r'([^\w\s{{}}])', '', regex=True)
        df.fillna("unknown", inplace=True)
        df['llama_split'] = df['llama_split'].apply(utils.normalize_senti_text)

        lang = file.split('/')[-1].split('.')[0]
        df['llama_label'] = df.apply(utils.llama_extract_senti_label, axis=1, args=(lang,))
        df = utils.filter_senti_labels(df)

        f1 = round(f1_score(df['label'], df['llama_label'], average='weighted') * 100, 2)

        language = utils.language_abv(lang)
        metric[language] = f1
    return metric


def ner_eval(prediction_files, metrics):
    for file in prediction_files:
        df = pd.read_csv(file, sep='\t', header=0)
        df['llama'] = df['llama'].str.lower()
        df['llama'] = df['llama'].str.replace('\n', ' ', regex=True)
        df['llama'] = df['llama'].str.replace('</s>', '', regex=True)
        df['llama'] = df['llama'].str.split('please').str[0].str.strip()
        df['llama'] = df['llama'].str.split('i hope').str[0].str.strip()

        df['llama'] = df.apply(utils.llama_extract_ner_pred, axis=1)

        df['target'] = df['target'].apply(utils.format_ner_text, target=True)
        df['llama'] = df['llama'].apply(utils.format_ner_text, target=False)
        df = df[~(df.target == '')]

        f1 = utils.calculate_ner_metrics(df, 'llama')
        language = file.split('/')[-1].split('.')[0]
        metrics[language] = f1
    return metrics


def mt_eval(prediction_files, metrics):
    for file in prediction_files:
        df = pd.read_csv(file, sep='\t')
        df['llama'] = df['llama'].str.lower()
        df['llama_split'] = df['llama'].str.replace('\n\n', ' ', regex=False)
        df['llama_split'] = df['llama_split'].str.replace('\n', ' ', regex=False)
        df['llama_split'] = df['llama_split'].str.replace('</s>', ' ', regex=False)
        df['llama_split'] = df['llama_split'].str.split('with that said').str[-1].str.strip()
        df['llama_split'] = df['llama_split'].str.split('with those limitations in mind').str[-1].str.strip()
        df['llama_split'] = df['llama_split'].str.split('with those considerations in mind').str[-1].str.strip()

        lang_full = file.split('/')[-1].split('.')[0]

        if lang_full.split('-')[1] == 'eng':
            lang = 'eng_Latn'
            language = 'english'
        elif lang_full.split('-')[1] == 'fra':
            lang = 'fra_Latn'
            language = 'french'
        elif lang_full.split('-')[1] == 'deu':
            lang = 'deu_Latn'
            language = 'german'
        else:
            lang = lang_full.split('-')[1]
            language = utils.lang_dict[lang].lower()

        df['llama_reponse'] = df.apply(utils.llama_extract_mt_pred, axis=1, args=(language,))
        df['llama_reponse'] = df['llama_reponse'].str.split('i hope this helps').str[0].str.strip()
        df['llama_reponse'] = df['llama_reponse'].str.split('i hope that helps').str[0].str.strip()
        df['llama_reponse'] = df['llama_reponse'].str.split('please note that').str[0].str.strip()

        df[[lang, 'llama_reponse']] = df[[lang, 'llama_reponse']].applymap(utils.normalize_text)

        lang_metric = utils.calculate_mt_metrics(df, 'llama_reponse', lang)
        metrics[lang_full] = lang_metric
    return metrics


def qa_eval(prediction_files, metrics):
    for file in prediction_files:
        df = pd.read_csv(file, sep='\t')
        df['translated_answer'] = df['answer_pivot'].apply(lambda x: x.split(': ')[-1].strip("['").rstrip("']}"))
        df['llama_response'] = df['llama'].str.lower()
        df['llama_response'] = df['llama_response'].str.split('information provided,').str[-1].str.strip()
        df['llama_response'] = df['llama_response'].str.split('information provided').str[-1].str.strip()
        df['llama_response'] = df['llama_response'].str.split('answer:').str[-1].str.strip()
        df['llama_response'] = df['llama_response'].str.split('\n').str[-1].str.strip()
        df['llama_response'] = df['llama_response'].str.replace('\n', ' ', regex=True)
        df['llama_response'] = df['llama_response'].str.replace('</s>', '', regex=False)

        df[['llama_response', 'translated_answer']] = df[['llama_response', 'translated_answer']].applymap(
            utils.normalize_text)
        df = df[~(df['translated_answer'] == '')]

        df['llama_response'] = df.apply(utils.check_yes_no, axis=1)

        language = file.split('/')[-1].split('.')[-2]

        lang_metric = utils.calculate_qa_metrics(df, 'llama_response')

        metrics[language] = lang_metric
    return metrics


def news_eval(prediction_files, metrics):
    for file in prediction_files:
        df = pd.read_csv(file, sep='\t')
        df['llama'] = df['llama'].str.lower()
        df['llama_split'] = df['llama'].str.replace('\n\n', ' ', regex=False)
        df['llama_split'] = df['llama_split'].str.replace('\n', ' ', regex=False)
        df['llama_split'] = df['llama_split'].str.replace('</s>', '', regex=False)

        df['llama_label'] = df.apply(utils.llama_extract_news_label, axis=1)
        df[['category', 'llama_label']] = df[['category', 'llama_label']].applymap(utils.normalize_text)

        # if it contains more than one label
        df['llama_label'] = df['llama_label'].apply(lambda x: "unknown" if x.count(' ') >= 1 else x)

        # assign random labels to unknowns
        df['llama_label'] = df.apply(utils.assign_label, axis=1, row_name='llama_label')

        f1 = round((f1_score(df['category'], df['llama_label'], average='weighted') * 100), 2)

        language = file.split('/')[-1].replace('.tsv', '')
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
    parser.add_argument('--task_function', type=str, default=qa_eval)
    parser.add_argument('--prediction_directory', type=str, default='../predictions/results_llama13b/qa')
    parser.add_argument('--output_directory', type=str, default='../results/llama13b/')

    args = parser.parse_known_args()
    main(args)
