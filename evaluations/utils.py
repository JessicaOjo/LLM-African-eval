import re
import os
import random
import evaluate
import collections
import numpy as np
import pandas as pd
from word2number import w2n

lang_dict = {'swa': 'Swahili',
             'tsn': 'Setswana',
             'mos': 'Mossi',
             'yor': 'Yoruba',
             'hau': 'Hausa',
             'fon': 'Fon',
             'pcm': 'Nigerian-Pidgin',
             'twi': 'Twi',
             'zul': 'Zulu',
             'bbj': 'Ghomala',
             'wol': 'Wolof',
             'bam': 'Bambara',
             'kin': 'Kinyarwanda',
             'xho': 'Xhosa',
             'lug': 'Luganda',
             'luo': 'Luo',
             'ewe': 'Ewe',
             'nya': 'Chichewa',
             'ibo': 'Igbo',
             'sna': 'chiShona',
             'amh': 'Amharic',
             'en': 'English',
             'fr': 'French',
             'eng': 'English',
             'orm': 'Oromo',
             'por-mz': 'Portuguese',
             'tir': 'Tigrinya',
             'tso': 'Tsonga',
             'arq': 'Algerian Arabic',
             'ary': 'Morrocan Arabic',
             }


def filter_senti_labels(df):
    # if label contains plurals
    plural_dict = {f'{label}s': label for label in ['positive', 'negative', 'neutral']}

    # Replacing the plurals with singular labels in the DataFrame
    df['llama_label'] = df['llama_label'].replace(plural_dict, regex=True)

    # if label is not one of 'positive', 'negative', 'neutral'
    df['llama_label'] = df['llama_label'].apply(lambda x: x if x in ['positive', 'negative', 'neutral'] else np.nan)

    # if it contains more than one label
    df['llama_label'] = df['llama_label'].apply(lambda x: np.nan if pd.notna(x) and x.count(' ') >= 1 else x)
    # Insert faux labels
    df.loc[df['llama_label'].isna(), 'llama_label'] = df['llama_label'].apply(
        lambda x: 'neutral' if x != 'neutral' else 'positive')

    return df


def filter_mt0_labels(df):
    df['mt0'] = df['mt0'].str.lower()
    # if label is not one of 'positive', 'negative', 'neutral'
    df['mt0'] = df['mt0'].apply(lambda x: x if x in ['positive', 'negative', 'neutral'] else np.nan)

    # if it contains more than one label
    df['mt0'] = df['mt0'].apply(lambda x: np.nan if pd.notna(x) and x.count(' ') >= 1 else x)
    # Insert faux labels
    df.loc[df['mt0'].isna(), 'mt0'] = df['mt0'].apply(lambda x: 'neutral' if x != 'neutral' else 'positive')

    return df


def verbalizer(value):
    verbalizer_dict = {
        "business": ['business', 'finance', 'economy', 'economics'],
        "entertainment": ['entertainment', 'music'],
        "health": ['health'],
        "politics": ['politics', 'world politics'],
        "religion": ['religion'],
        "sports": ['sports'],
        "technology": ['technology', 'tech']}
    for key, values in verbalizer_dict.items():
        for v in values:
            if v in value:
                return key
    return value


def assign_label(row, row_name):
    labels = ["business", "entertainment", "health", "politics", "religion", "sports", "technology"]
    if row[row_name] not in labels:
        new_labels = [label for label in labels if label != row['category']]
        return random.choice(new_labels)
    return row[row_name]


def format_ner_text(text, target=False):
    label_dict = {'person': 'PER',
                  'location': 'LOC',
                  'organization': 'ORG'}
    text = text.lower()
    for key, value in label_dict.items():
        text = text.replace(key, value) if not target else text.replace(value.lower(), value)

    if not target:
        text = "$$".join(i if 'date' not in i else '' for i in text.split('$'))

        return text.rstrip('$$')

    text = "$".join(i if 'date' not in i else '' for i in text.split('$$'))
    return text.rstrip('$$')


def span_f1_seqio(targets, predictions):
    """Computes Span based F1 score.

    This function is copied from
    https://github.com/google-research/multilingual-t5/blob/master/multilingual_t5/evaluation/metrics.py

    Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings

    Returns:
    span f1 across all targets and predictions (Based on CoNLL script)
    """
    true_positives = collections.defaultdict(int)
    false_positives = collections.defaultdict(int)
    false_negatives = collections.defaultdict(int)

    def tags_to_spans(tag_sequence, delimiter="$$"):
        """Extract spans from IOB1 or BIO tags."""
        tag_sequence_split = [item.strip() for sub in tag_sequence.split('$$') for item in sub.split('$') if item]
        tags_entities = []
        for tag_entity in tag_sequence_split:
            tag_entity_split = tag_entity.split(":")
            if len(tag_entity_split) != 2:
                continue
            tag = normalize_text(tag_entity_split[0].strip())
            entity = normalize_text(tag_entity_split[1].strip())
            tags_entities.append((tag, entity))
        return tags_entities

    def compute_f1_metrics(true_positives, false_positives, false_negatives):
        precision = float(true_positives) / float(
            true_positives + false_positives + 1e-13
        )
        recall = float(true_positives) / float(
            true_positives + false_negatives + 1e-13
        )
        f1_measure = 2.0 * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    for target, pred in zip(targets, predictions):
        gold_spans = tags_to_spans(target)
        predicted_spans = tags_to_spans(pred)

        for span in predicted_spans:
            if span in gold_spans:
                true_positives[span[0]] += 1
                gold_spans.remove(span)
            else:
                false_positives[span[0]] += 1
        # These spans weren't predicted.
        for span in gold_spans:
            false_negatives[span[0]] += 1

        _, _, f1_measure = compute_f1_metrics(
            sum(true_positives.values()),
            sum(false_positives.values()),
            sum(false_negatives.values()),
        )

    return {"span_f1": f1_measure}


def normalize_text(string):
    def get_blank_spaces_pattern():
        return re.compile(r'\s{3,}|\t')

    def remove_blank_spaces(text):
        text = re.sub(pattern=get_blank_spaces_pattern(), repl='', string=text)
        text = re.sub('\s+', ' ', text)
        return text

    def remove_punctuation(text):
        my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@.""-,`'
        text = re.sub('[' + my_punctuation + ']+', ' ', str(text))  # strip punctuation
        return text

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def lowercase(text):
        text = text.lower()
        return text

    string = remove_punctuation(string)
    string = remove_articles(string)
    string = remove_blank_spaces(string)
    string = lowercase(string)

    return string


def normalize_senti_text(string):
    def get_blank_spaces_pattern():
        return re.compile(r'\s{3,}|\t')

    def remove_blank_spaces(text):
        text = re.sub(pattern=get_blank_spaces_pattern(), repl='', string=text)
        text = re.sub('\s+', ' ', text)
        return text

    def lowercase(text):
        text = text.lower()
        return text

    string = remove_blank_spaces(string)
    string = lowercase(string)

    return string


def llama_extract_senti_label(row, lang):
    if row['llama_split'].startswith("i apologize") or row['llama_split'].startswith("im just an ai"):
        return np.nan

    match = re.search(r'{{(.+?)}}', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'label it as having a (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'label it as (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'label the sentiment as (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'sentiment can be labeled (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'categorize it as (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'tone remains (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'labeled as (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'as sentiment (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'statement has a (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'has a\s+\{(\w+)\}', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'has a (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'sentiment as (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'statement as having a (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'{} statement as (\w+)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'provided as (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'statement is (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'statement as (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'suggests a (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'interpreted as having a (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'interpreted as (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'as a (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'indicate a (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'appears to be a (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'considered a (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'as (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'sentence is (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'provided is (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'is (\w+)', row['llama_split'])
    if match:
        return match.group(1)

    return np.nan


def llama_extract_ner_pred(row):
    if row['llama'].startswith("i apologize") or row['llama'].startswith("i'm just an ai") or row['llama'].startswith(
            "i cannot"):
        return row['llama']

    match = re.search(r'provided:\s+(.*)', row['llama'])
    if match:
        return match.group(1)

    match = re.search(r'passage:\s+(.*)', row['llama'])
    if match:
        return match.group(1)

    match = re.search(r'are:\s+(.*)', row['llama'])
    if match:
        return match.group(1)

    match = re.search(r'separator:\s+(.*)', row['llama'])
    if match:
        return match.group(1)

    match = re.search(r'entities:\s+(.*)', row['llama'])
    if match:
        return match.group(1)

    return row['llama']


def llama_extract_mt_pred(row, lang):
    if row['llama_split'].startswith("i apologize") or row['llama_split'].startswith("i'm just an ai") or row[
        'llama_split'].startswith("i cannot") or row['llama_split'].startswith("i can't"):
        return row['llama_split']

    if "in fon" in row['llama_split']:
        match = re.search(r'be:\s+"(.*?)"', row['llama_split'])
        if match:
            return match.group(1)

    match = re.search(r'{} sentence:\s+(.*)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'translation:\s+(.*)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'translated to {} as:\s+(.*)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'translated to {} as\s+(.*)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'{} as:\s+"(.*?)"'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'{} as:\s+(.*?)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'translated sentence:\s+(.*)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'translates to\s+(.*)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'{} as follows:\s+"(.*?)"'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'{} as follows:\s+(.*)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'into {}:\s+(.*)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'to {}:\s+(.*)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'{}:\s+(.*)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'{} is:\s+(.*)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'{} sentence is:\s+(.*)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'{} sentence:\s+(.*)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'is:\s+(.*)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'translated as:\s+(.*)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'would be:\s+(.*)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'translates to:\s+"(.*?)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'translates to:\s+(.*?)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'{} as\s+"(.*?)"'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'{} is\s+"(.*?)"'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'{} is\s+(.*?)'.format(lang), row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'translates to\s+"(.*?)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'translates to\s+(.*)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(':\s+"(.*?)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(':\s+(.*?)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'sentence\s+(.*)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'sentence is\s+(.*)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'this means\s+(.*)', row['llama_split'])
    if match:
        return match.group(1)

    return row['llama_split']


def llama_extract_news_label(row):
    if row['llama_split'].startswith("i apologize") or row['llama_split'].startswith("i'm just an ai") or row[
        'llama_split'].startswith("i cannot"):
        return row['llama_split']  # np.nan

    if row['llama_split'].endswith("category."):
        row['llama_split'] = row['llama_split'].rstrip("category.")

    if row['llama_split'].startswith("labels:\s+(.*?)"):
        row['llama_split'] = row['llama_split'].str.split("labels:\s+(.*?)")[0]

    match = re.search(r'labels:\s+"(\w+)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'labels:\s+\*\s+(\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'labels:\s+(\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'provided:\s+\*\s+(\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'provided:\s+\{\{\"(\w+)\"\}\}', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'provided:\s+(\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'related to:\s+\{\{\"(\w+)\"\}\}', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'related to\s+(\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'text:\s+\{\{\"(.*?)\"\}\}', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'label:\s+(\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'news as\s+"(\w+)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'could be\s+"(\w+)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'under\s+"(\w+)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'would be:\s+\{\{\"(.*?)\"\}\}\}', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'text:\s+\*\s+(\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'regarding\s+\{\"(\w+)\"\}', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'regarding\s+"(\w+)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'regarding the category of\s+"(.*?)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'regarding the category of\s+(.*?)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'regarding the category\s+"(\w+)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'regarding the category\s+(\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'regarding the\s+"(\w+)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'regarding the\s+(\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'regarding\s+"(\w+)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'regarding\s+(\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'category is\s+"(.*?)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'category is\s+(.*?)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r':\s+\*\s+(\w+)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'is\s+"(.*?)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'would be\s+"(.*?)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'the\s+"(.*?)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'to\s+"(.*?)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'this as\s+"(.*?)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'label it as\s+"(.*?)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'label as\s+"(.*?)":', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'related to\s+(.*?)', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r'as\s+"(.*?)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r':\s+\*\*(\w+)\*\*', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r':\s+"(.*?)"', row['llama_split'])
    if match:
        return match.group(1)

    match = re.search(r':\s+(.*?)', row['llama_split'])
    if match:
        return match.group(1)

        match = re.search(r'category of\s+"(.*?)"', row['llama_split'])
    if match and row['llama_split'].endswith("category.") == False:
        return match.group(1)

    match = re.search(r'category\s+"(.*?)"', row['llama_split'])
    if match and row['llama_split'].endswith("category.") == False:
        return match.group(1)

    match = re.search(r'category\s+(.*?)', row['llama_split'])
    if match and row['llama_split'].endswith("category.") == False:
        return match.group(1)

    return row['llama_split']


def contains_number(text, number_word):
    try:
        number = w2n.word_to_num(number_word)
    except ValueError:
        return False

    return str(number) in text


def check_yes_no(row):
    has_number = contains_number(row['translated_answer'], row['llama_response'])
    if has_number:
        return row['translated_answer']

    if row['translated_answer']:
        match = re.search(r'\b(?:no|yes)\b', row['llama_response'])
        if match:
            return match.group(0)

    match = re.search(r'\b{}\b'.format(row['translated_answer']), row['llama_response'])
    if match:
        return match.group(0)

    return row['llama_response']


def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def gpt4_match_news_columns(df):
    """ Extracts category from GPT4 results with more than one categories"""

    labels = ["business", "entertainment", "health", "politics", "religion", "sports", "technology"]

    df['gpt-4'] = df['gpt-4'].str.lower()
    df['gpt-4-clean'] = 'unknown'

    for index, row in df.iterrows():
        if not row['gpt-4'].startswith('this') and not row['gpt-4'].startswith('the'):
            if row['gpt-4'].startswith('"') and row['gpt-4'].endswith('"'):
                df.loc[index, 'gpt-4'] = row['gpt-4'].strip('"')
            else:
                row['gpt-4'] = row['gpt-4'].split(', ')
                df.loc[index, 'gpt-4-clean'] = row['gpt-4'][0]

        else:
            for label in labels:
                df['gpt-4-clean'][(df['gpt-4-clean'] == 'unknown') &
                                  (df['gpt-4'].str.contains('does not') == False) &
                                  (df['gpt-4'].str.contains("doesn't") == False) &
                                  (df['gpt-4'].str.contains("not in english") == False) &
                                  (df['gpt-4'].str.contains("is not english") == False) &
                                  (df['gpt-4'].str.contains("cannot") == False) &
                                  (df['gpt-4'].str.contains(label.lower()))] = label

    return df


def language_abv(language):
    for key, value in lang_dict.items():
        if language == value:
            return key


def calculate_qa_metrics(df, row_name):
    pred_squad = []
    ref_squad = []
    for index, row in df.iterrows():
        pred_dict = {'prediction_text': row[row_name], 'id': str(index)}
        ref_dict = {'answers': {'answer_start': [1], 'text': [row['translated_answer']]}, 'id': str(index)}
        pred_squad.append(pred_dict)
        ref_squad.append(ref_dict)

    pred_em = [i for i in df[row_name]]
    ref_em = [i for i in df['translated_answer']]

    em = evaluate.load('exact_match')
    squad_metric = evaluate.load("squad")

    results_squad = squad_metric.compute(predictions=pred_squad, references=ref_squad)
    results_em = em.compute(predictions=pred_em, references=ref_em)

    lang_metric = {'f1_squad': round(results_squad['f1'], 1), 'em': round(results_em['exact_match'], 1)}

    return lang_metric


def calculate_ner_metrics(df, row_name):
    pred = [i for i in df[row_name]]
    ref = [i for i in df['target']]
    f1 = round((span_f1_seqio(ref, pred)['span_f1'] * 100), 2)
    return f1


def calculate_mt_metrics(df, row_name, lang):
    lang_metric = {}
    pred = [str(i) for i in df[row_name]]
    ref = [str(i) for i in df[lang]]
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    lang_metric['chrf'] = round(chrf.compute(predictions=pred, references=ref)["score"], 2)
    lang_metric['bleu'] = round(bleu.compute(predictions=pred, references=ref)["score"], 2)
    return lang_metric