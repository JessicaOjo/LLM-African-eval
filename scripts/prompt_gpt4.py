from __future__ import annotations
import os

import glob
import json
import openai
import asyncio
import logging
import aiolimiter
import pandas as pd
from pathlib import Path
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio

from typing import Any
from typing import List

from sklearn.metrics import f1_score


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


async def dispatch_openai_requests(
        messages_list: List[List[dict[str, str]]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
) -> List[str]:
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


async def _throttled_openai_chat_completion_acreate(
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(100):
            try:
                return await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                logging.warning("OpenAI API rate limit exceeded. Sleeping for 50 seconds.")
                await asyncio.sleep(50)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 50 seconds.")
                await asyncio.sleep(50)
            except openai.error.ServiceUnavailableError:
                logging.warning("OpenAI Server overload. Sleeping for 1 minute.")
                await asyncio.sleep(60)
            except openai.error.APIConnectionError:
                logging.warning("OpenAI Communication Error. Sleeping for 2 minutes.")
                await asyncio.sleep(120)
            except openai.error.Timeout:
                logging.warning("OpenAI Timeout. Sleeping for 2 minutes.")
                await asyncio.sleep(120)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
        full_contexts: List[List[dict[str, str]]],
        model_config: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    session = ClientSession()
    openai.aiosession.set(session)
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model_config,
            messages=full_context,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for full_context in full_contexts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    await session.close()
    return [x["choices"][0]["message"]["content"] for x in responses]


def getlabel_string(filename):
    with open(filename) as f:
        label_list = f.read().splitlines()
    label_string = label_list[0]
    for i, value in enumerate(label_list[:-2], 1):
        label_string += ', ' + label_list[i]

    label_string += ' or ' + label_list[-1]

    return label_string, label_list


def get_language(files, senti=False, mt=False):
    if senti:
        lang = [i.split('/')[-2] for i in files]
        languages = ['Tigrinya', 'Amharic', 'Igbo', 'Oromo', 'Kinyarwanda', 'Portuguese', 'Pidgin', 'Twi', 'Tsonga',
                     'Arya',
                     'Yoruba', 'Hausa', 'Arabic', 'Swahili']
        return dict(zip(lang, languages))
    if mt:
        languages = ['Yoruba', 'Zulu', 'Hausa', 'Setswana', 'Swahili', 'Nigerian-Pidgin', 'Fon', 'Twi', 'Mossi',
                     'Ghomala',
                     'Wolof', 'Luganda', 'Chichewa', 'Bambara', 'Kinyarwanda', 'Ewe', 'Xhosa', 'Igbo', 'Amharic',
                     'Shona']
        lang = [i.split('/')[-2].split('-')[1] for i in files]
        return dict(zip(lang, languages))


def sentiment():
    """ Identifies tweet sentiments for different languages"""

    files = glob.glob('afrisenti/data/**/test.tsv', recursive=True)
    languages = get_language(files, senti=True)
    label = '{{Neutral, Positive or Negative}}'
    metric = {}
    for file in files:
        language = file.split('/')[-2]
        if language == 'eng':
            df = pd.read_csv(file, sep='\t', header=0)
            language = file.split('/')[-2]
            language = [v for k, v in languages.items() if k == language][0]
            print(f'\nLanguage: {language}')
            all_input_messages = []
            for index in range(df.shape[0]):
                text = df['text'].iloc[index]
                message = f'Does this {language} statement; "{text}" have a {label} sentiment? Labels only'
                input_mes = [{"role": "system", "content": "You are an assistant able to detect sentiments in tweets"},
                             {"role": "user", "content": message[:500]}]
                all_input_messages.append(input_mes)

            responses = asyncio.run(generate_from_openai_chat_completion(full_contexts=all_input_messages,
                                                                         model_config=model_name,
                                                                         temperature=0.3,
                                                                         max_tokens=500,
                                                                         top_p=1.0))
            completions = []
            for completion_text in responses:
                completions.append(completion_text.lower())
            df['gpt-4'] = completions
            df.to_csv(output_dir + language + '.tsv', sep='\t')

            f1 = f1_score(df['label'], df['gpt-turbo'], average='weighted')
            metric[language] = f1

    with open('sentiment_gpt4_eng.json', 'w') as outfile:
        json.dump(metric, outfile)


def news_classification(reduced=False):
    files = glob.glob('masakhane-news/data/**/test.tsv', recursive=True)
    if reduced:
        files = glob.glob('masakhane-news/reduced_data/**/test.tsv', recursive=True)
    prompt_prefix = 'Is this a piece of news regarding {{"'
    prompt_suffix = '"}}? '
    metric = {}

    for file in files:
        file_path = Path(file)
        df = pd.read_csv(file, sep='\t')
        if reduced:
            label_string, label_list = "health or politics or sports", ['health', 'politics', 'sports']
        else:
            label_string, label_list = getlabel_string(Path(f'{file_path.parent}/labels.txt'))

        lang = file.split('/')[-2]

        all_input_messages = []
        for index in range(df.shape[0]):  # df.shape[0]
            headline = df['headline'].iloc[index]
            content = df['text'].iloc[index]
            text_string = headline + ' ' + content
            query = ' '.join(text_string.split()[:100])

            message = 'Labels only. ' + prompt_prefix + label_string + prompt_suffix + query
            input_mes = [{"role": "system", "content": "You are an assistant able to categorize news articles"},
                         {"role": "user", "content": message[:500]}]
            all_input_messages.append(input_mes)

        responses = asyncio.run(
            generate_from_openai_chat_completion(all_input_messages, model_name, 0.3, 500, 1.0, 500))

        completions = []
        for completion_text in responses:
            completions.append(completion_text)
        df['gpt-4'] = completions
        df.to_csv(output_dir + lang + '.tsv', sep='\t')
        for label in label_list:
            df['gpt-4'][df['gpt-4'].str.contains(label.lower())] = label

        f1 = f1_score(df['category'], df['gpt-4'], average='weighted')
        metric[lang] = f1

    with open('masakhane-news_gpt4.json', 'w') as outfile:
        json.dump(metric, outfile)


def cross_lingual_qa(pivot=False):
    languages = ['ibo', 'bem', 'kin', 'twi', 'fon', 'zul', 'yor', 'hau', 'swa']
    for language in languages:
        gold_passages = glob.glob(f'afriqa/data/gold_passages/{language}/*test.json')
        gp_df = pd.read_json(gold_passages[0], lines=True)

        pivot_lang = "French" if gold_passages[0].split('.')[-3] == 'fr' else "English"
        prompt_query = f"Use the following pieces of context to answer the provided question. If you don't know the answer, \
just say that you don't know, don't try to make up an answer. Provide the answer with the least number of \
words possible. Provide the answer only. Provide answer in {pivot_lang}. Do not repeat the question"

        all_input_messages = []
        for index in range(len(gp_df)):
            context = f"Context: {gp_df['context'].iloc[index]}"
            question = f"Question: {gp_df['question_translated'].iloc[index]}" if pivot else f"Question: {gp_df['question_lang'].iloc[index]}"
            message = prompt_query + '\n\n' + context + '\n' + question
            input_mes = [{"role": "system", "content": "You are an assistant able to retrieve answers from a passage"},
                         {"role": "user", "content": message}]
            all_input_messages.append(input_mes)

        responses = asyncio.run(generate_from_openai_chat_completion(full_contexts=all_input_messages,
                                                                     model_config=model_name,
                                                                     temperature=0.3,
                                                                     max_tokens=500,
                                                                     top_p=1.0, ))
        completions = []
        for completion_text in responses:
            completions.append(completion_text.lower())
        gp_df['gpt-4'] = completions
        gp_df.to_csv(output_dir + language + '.tsv', sep='\t')


def machine_translation(reverse=False):
    files = glob.glob('lafand-mt/data/tsv_files/**/test.tsv', recursive=True)
    languages = get_language(files, mt=True)

    for file in files:
        if file.split('/')[-2].split('-')[1] == 'ewe':
            df = pd.read_csv(file, sep='\t', header=0)
            pivot_lang_abv, target_lang_abv = file.split('/')[-2].split('-')[0], file.split('/')[-2].split('-')[1]
            target_lang = [v for k, v in languages.items() if k == target_lang_abv][0]
            pivot_lang = 'English' if pivot_lang_abv == 'en' else 'French'

            print(f'language: {target_lang}')
            all_input_messages = []
            for index in range(df.shape[0]):
                if not reverse:
                    text = df['en'].iloc[index] if pivot_lang_abv == 'en' else df['fr'].iloc[index]
                    prompt_query = f"Translate the {pivot_lang} sentence below to {target_lang}. Return the translated \
                    sentence only. If you cannot translate the sentence simply say you don't know"
                else:
                    text = df[target_lang_abv].iloc[index]
                    prompt_query = f"Translate the {target_lang} sentence below to {pivot_lang}. Return the translated \
                                    sentence only. If you cannot translate the sentence simply say you don't know"

                message = prompt_query + '\n' + text
                input_mes = [{"role": "system", "content": "You are an assistant able to translate texts"},
                             {"role": "user", "content": message}]
                all_input_messages.append(input_mes)

            responses = asyncio.run(generate_from_openai_chat_completion(full_contexts=all_input_messages,
                                                                         model_config=model_name,
                                                                         temperature=0.3,
                                                                         max_tokens=500,
                                                                         top_p=1.0))
            completions = []
            for completion_text in responses:
                completions.append(completion_text.lower())
            df['gpt-4'] = completions
            if reverse:
                df.to_csv(output_dir + f'{target_lang_abv}-{pivot_lang_abv}' + '.tsv', sep='\t')
            else:
                df.to_csv(output_dir + f'{pivot_lang_abv}-{target_lang_abv}' + '.tsv', sep='\t')


def machine_translation_germany(reverse=False):
    files = glob.glob('ntrex/test/*.tsv', recursive=True)
    languages = {"deu": "Deutsch",
                 "eng": "English",
                 "fra": "French"}

    for file in files:
        df = pd.read_csv(file, sep='\t', header=0)
        pivot_lang_abv, target_lang_abv = file.split('/')[-1].split('_')[2], file.split('/')[-1].split('_')[0]
        target_lang = [v for k, v in languages.items() if k == target_lang_abv][0]
        pivot_lang = 'English' if pivot_lang_abv == 'eng' else 'French'

        print(f'target language: {target_lang}, {target_lang_abv}')
        print(f'pivot language: {pivot_lang}, {pivot_lang_abv}')

        all_input_messages = []
        for index in range(df.shape[0]):
            if not reverse:
                text = df['eng_Latn'].iloc[index] if pivot_lang_abv == 'eng' else df['fra_Latn'].iloc[index]
                prompt_query = f"Translate the {pivot_lang} sentence below to {target_lang}. Return the translated sentence only. If you cannot translate the sentence simply say you don't know"
            else:
                text = df['deu_Latn'].iloc[index]
                prompt_query = f"Translate the {target_lang} sentence below to {pivot_lang}. Return the translated sentence only. If you cannot translate the sentence simply say you don't know"

            message = prompt_query + '\n' + text
            input_mes = [{"role": "system", "content": "You are an assistant able to translate texts"},
                         {"role": "user", "content": message}]
            all_input_messages.append(input_mes)

        responses = asyncio.run(generate_from_openai_chat_completion(full_contexts=all_input_messages,
                                                                     model_config=model_name,
                                                                     temperature=0.3,
                                                                     max_tokens=500,
                                                                     top_p=1.0, ))
        completions = []
        for completion_text in responses:
            completions.append(completion_text.lower())
        df['gpt-4'] = completions
        if reverse:
            df.to_csv(output_dir + f'{target_lang_abv}-{pivot_lang_abv}' + '.tsv', sep='\t')
        else:
            df.to_csv(output_dir + f'{pivot_lang_abv}-{target_lang_abv}' + '.tsv', sep='\t')


def named_entity_recognition():
    prompt_query = "Named entities refers to names of location, organisation and personal name. \n\
For example, 'David is an employee of Amazon and he is visiting New York next week to see Esther' will be \n\
PERSON: David $ ORGANIZATION: Amazon $ LOCATION: New York $ PERSON: Esther \n\n\
List all the named entities in the passage above using $ as separator. Return only the output"

    files = glob.glob('masakhane-ner/xtreme-up/MasakhaNER-X/test/*.jsonl', recursive=True)

    for file in files:
        with open(file) as data:
            data_lines = data.read().splitlines()

        data_dicts = [json.loads(line) for line in data_lines]
        df = pd.DataFrame(data_dicts)
        df = df[~(df['target'] == '')]

        all_input_messages = []
        for index in range(df.shape[0]):
            text = df['text'].iloc[index]
            message = text + '\n\n' + prompt_query
            input_mes = [{"role": "system", "content": "You are an assistant able to identify named entities"},
                         {"role": "user", "content": message}]
            all_input_messages.append(input_mes)

        responses = asyncio.run(generate_from_openai_chat_completion(full_contexts=all_input_messages,
                                                                     model_config=model_name,
                                                                     temperature=0.3,
                                                                     max_tokens=500,
                                                                     top_p=1.0, ))
        completions = []
        for completion_text in responses:
            completions.append(completion_text.lower())
        df['gpt-4'] = completions
        file_lang = file.split('/')[-1].split('.')[0]
        df.to_csv(output_dir + file_lang + '.tsv', sep='\t')


def named_entity_recognition_new():
    prompt_query = "Task Description: Your task is to identify and label any named entities present " \
                   "in the text. The named entity labels that you will be using are PER (person), LOC (location) and " \
                   "ORG (organization). You may encounter multi-word entities, so make sure to label each word of the " \
                   "entity with the appropriate prefix ('B' for the first word of the entity, 'I' for any non-initial " \
                   "word of the entity). For words which are not part of any named entity, you should return 'O'. \n\n" \
                   "Note: Your output format should be a list of tuples, where each tuple consists of a word from the " \
                   "input text and its corresponding named entity label."

    files = glob.glob('masakhane-ner/xtreme-up/MasakhaNER-X/test/*.jsonl', recursive=True)

    for file in files:
        with open(file) as data:
            data_lines = data.read().splitlines()

        data_dicts = [json.loads(line) for line in data_lines]
        df = pd.DataFrame(data_dicts)
        df = df[~(df['target'] == '')]

        all_input_messages = []
        for index in range(df.shape[0]):
            text = df['text'].iloc[index].split()
            message = prompt_query + "\n\n" + f"Input: {text}"

            input_mes = [{"role": "system", "content": "You are working as a named entity recognition expert and your "
                                                       "task is to label a given text with named entity labels."},
                         {"role": "user", "content": message}]
            all_input_messages.append(input_mes)

        responses = asyncio.run(generate_from_openai_chat_completion(full_contexts=all_input_messages,
                                                                     model_config=model_name,
                                                                     temperature=0.3,
                                                                     max_tokens=500,
                                                                     top_p=1.0, ))
        completions = []
        for completion_text in responses:
            completions.append(completion_text.lower())
        df['gpt-4'] = completions
        file_lang = file.split('/')[-1].split('.')[0]
        print(file_lang)
        df.to_csv(output_dir + file_lang + '.tsv', sep='\t')


def summarization():
    african_languages = ['yoruba', 'somali', 'portuguese', 'tigrinya', 'kirundi', 'igbo', 'amharic',
                         'oromo', 'pidgin', 'swahili', 'arabic', 'hausa', 'english']
    files = glob.glob('XLSum_complete_v2.0/*test.jsonl', recursive=True)
    for file in files:
        file_lang = file.split('/')[-1].split('.')[0].split('_')[0]
        if file_lang in african_languages:
            with open(file) as f:
                lines = f.read().splitlines()

            line_dicts = [json.loads(line) for line in lines]
            df = pd.DataFrame(line_dicts)
            prompt_query = f"In the least amount of sentences, summarise this {file_lang} passage. The summary should be consistent with the language of the passage."
            all_input_messages = []
            for index in range(df.shape[0]):
                text = df['text'].iloc[index]
                message = prompt_query + '\n\n' + text
                input_mes = [{"role": "system", "content": "You are an assistant able to summarize passages"},
                             {"role": "user", "content": message}]
                all_input_messages.append(input_mes)

            responses = asyncio.run(generate_from_openai_chat_completion(full_contexts=all_input_messages,
                                                                         model_config=model_name,
                                                                         temperature=0.3,
                                                                         max_tokens=500,
                                                                         top_p=1.0, ))
            completions = []
            for completion_text in responses:
                completions.append(completion_text.lower())
            df['gpt-4'] = completions
            df.to_csv(output_dir + file_lang + '.tsv', sep='\t')


def main(senti: bool = False,
         news: bool = False,
         qa: bool = False,
         mt: bool = False,
         mtg: bool = False,
         ner: bool = False,
         nern: bool = False,
         summ: bool = False):
    """ Runs the task functions"""
    if senti is True:
        sentiment()
    elif news is True:
        news_classification(reduced=True)
    elif qa is True:
        cross_lingual_qa(pivot=True)
    elif mt is True:
        machine_translation(reverse=False)
    elif mtg is True:
        machine_translation_germany(reverse=True)
    elif ner is True:
        named_entity_recognition()
    elif nern is True:
        named_entity_recognition_new()
    elif summ is True:
        summarization()


if __name__ == '__main__':
    openai.api_key = os.getenv('open_ai_key')
    model_name = 'gpt-4-0613'
    output_dir = 'gpt_4_results/masakhaNER/new_prompt/'  # change based on task
    create_dir(output_dir)

    main(nern=True)  # toggle based on task
