from __future__ import annotations
import os


import glob
import json
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch



from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

def load_llama(model_name):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    import transformers
    import torch

    model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-xxl", cache_dir="/SAN/intelsys/llm/dadelani/models")

    tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-xxl")

    model = model.to(device)

    return model, tokenizer


'''
def prompt_llm(model, tokenizer, text):

    
    sequences = pipeline(text,
                         do_sample=True,
                         top_k=10,
                         num_return_sequences=1,
                         eos_token_id=tokenizer.eos_token_id,
                         max_length=256, device=device
                         )
    

    #inputs = tokenizer.encode(text, return_tensors="pt")
    #outputs = model.generate(inputs)
    #result = tokenizer.decode(outputs[0])

    inputs = tokenizer(text, return_tensors="pt").to(device)
    generate_ids = model.generate(inputs.input_ids, max_length=512)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    #.input_ids.to(device)

    #result = ' '.join([seq['generated_text'] for seq in sequences])

    return result
'''
# script from Nikita Vassilyev and Alex Pejovic
def prompt_llm(
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        message: str,
        temperature: float = 0.7,
        repetition_penalty: float = 1.176,
        top_p: float = 0.1,
        top_k: int = 40,
        num_beams: int = 1,
        max_new_tokens: int = 512,
):
    inputs = tokenizer(message, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)

    if input_ids.shape[-1] > 4096:
        print("Input too long, exceeds 4096 tokens")
        return None

    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig

    generation_config = GenerationConfig(
        ### temperature, top_p, and top_k are not needed since we are using 1 beam
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )

    #result = tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output

from sklearn.metrics import f1_score


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def getlabel_string(filename):
    with open(filename) as f:
        label_list = f.read().splitlines()
    label_string = label_list[0]
    for i, value in enumerate(label_list[:-2], 1):
        label_string += ', ' + label_list[i]

    label_string += ' or '+label_list[-1]

    return label_string, label_list


def get_language(files, senti=False, mt=False):
    if senti:
        lang = sorted([i.split('/')[-2] for i in files])
        languages = ['Amharic', 'Algerian Arabic', 'Morrocan Arabic', 'English', 'Hausa', 'Igbo', 'Kinyarwanda', 'Oromo',
                     'Nigerian Pidgin', 'Portuguese', 'Swahili', 'Tigrinya', 'Tsonga', 'Twi', 'Yoruba']
        return dict(zip(lang, languages))
    if mt:
        languages = ['Yoruba', 'Zulu', 'Hausa', 'Setswana', 'Swahili', 'Nigerian-Pidgin', 'Fon', 'Twi', 'Mossi', 'Ghomala',
                     'Wolof', 'Luganda', 'Chichewa', 'Bambara', 'Kinyarwanda', 'Luo', 'Ewe', 'Xhosa', 'Igbo', 'Amharic', 'Shona']
        lang = [i.split('/')[-2].split('-')[1] for i in files]
        return dict(zip(lang, languages))


def sentiment(model_pipeline, tokenizer, output_dir):
    """ Identifies tweet sentiments for different languages"""

    files = glob.glob('afrisenti/data/**/test.tsv', recursive=True)
    print(files)
    languages = get_language(files, senti=True)
    label = '{{Neutral, Positive or Negative}}'
    metric = {}
    for file in files:
        #language = file.split('/')[-2]
        #if language == 'eng':
        df = pd.read_csv(file, sep='\t', header=0)
        language = file.split('/')[-2]
        language = [v for k, v in languages.items() if k == language][0]
        print(f'\nLanguage: {language}')
        responses = []
        for index in range(df.shape[0]):
            text = df['text'].iloc[index]
            message = f'Does this {language} statement; "{text}" have a {label} sentiment? Labels only'
            input_mes = message

            result = prompt_llm(model_pipeline, tokenizer, input_mes)
            responses.append(result)

            if index%100==0:
                print(index, 'processed', result)

        completions = []
        for completion_text in responses:
            completions.append(completion_text)


        df['mt0'] = completions
        df.to_csv(output_dir + language + '.tsv', sep='\t')

        #f1 = f1_score(df['label'], df['mt0'], average='weighted')
        #metric[language] = f1

    #with open('sentiment_mt0.json', 'w') as outfile:
    #    json.dump(metric, outfile)

def news_classification(model_pipeline, tokenizer, output_dir):
    files = glob.glob('masakhane-news/data/**/test.tsv', recursive=True)
    prompt_prefix = 'Is this a piece of news regarding {{"'
    prompt_suffix = '"}}? '
    metric = {}

    for file in files:
        file_path = Path(file)
        df = pd.read_csv(file, sep='\t')
        label_string, label_list = getlabel_string(Path(f'{file_path.parent}/labels.txt'))
        lang = file.split('/')[-2]

        responses = []
        for index in range(df.shape[0]): #df.shape[0]
            headline = df['headline'].iloc[index]
            content = df['text'].iloc[index]
            text_string = headline + ' ' + content
            query = ' '.join(text_string.split()[:100])

            message = 'Labels only. ' + prompt_prefix + label_string + prompt_suffix + query

            input_mes =  message

            result = prompt_llm(model_pipeline, tokenizer, input_mes)
            responses.append(result)


            if index%100==0:
                print(index, 'processed', result)


        completions = []
        for completion_text in responses:
            completions.append(completion_text)
        df['mt0'] = completions
        df.to_csv(output_dir+lang+'.tsv', sep='\t')
        #for label in label_list:
        #    df['mt0'][df['mt0'].str.contains(label.lower())] = label

        #f1 = f1_score(df['category'], df['mt0'], average='weighted')
        #metric[lang] = f1

    #with open('masakhane-news_gpt4.json', 'w') as outfile:
    #    json.dump(metric, outfile)


def cross_lingual_qa(model_pipeline, tokenizer, output_dir, pivot=False):
    languages = ['ibo', 'bem', 'kin', 'twi', 'fon', 'zul', 'yor', 'hau', 'swa']
    for language in languages:
        print(language)
        gold_passages = glob.glob(f'afriqa/data/gold_passages/{language}/*test.json')
        gp_df = pd.read_json(gold_passages[0], lines=True)

        pivot_lang = "French" if gold_passages[0].split('.')[-3] == 'fr' else "English"
        prompt_query = f"Use the following pieces of context to answer the provided question. If you don't know the answer, \
just say that you don't know, don't try to make up an answer. Provide the answer with the least number of \
words possible. Provide the answer only. Provide answer in {pivot_lang}. Do not repeat the question"

        responses = []
        for index in range(len(gp_df)):
            context = f"Context: {gp_df['context'].iloc[index]}"
            question = f"Question: {gp_df['question_translated'].iloc[index]}" if pivot else f"Question: {gp_df['question_lang'].iloc[index]}"
            message = prompt_query + '\n\n' + context + '\n' + question


            input_mes = message

            result = prompt_llm(model_pipeline, tokenizer, input_mes)
            responses.append(result)

            if index % 100 == 0:
                print(index, 'processed', result)

        #print(responses[:10])
        completions = []
        for completion_text in responses:
            completions.append(completion_text)
        gp_df['mt0'] = completions
        gp_df.to_csv(output_dir + language + '.tsv', sep='\t')


def machine_translation(model_pipeline, tokenizer, output_dir, reverse=False):
    files = glob.glob('lafand-mt/data/tsv_files/**/test.tsv', recursive=True)
    languages = get_language(files, mt=True)

    for file in files:
        df = pd.read_csv(file, sep='\t', header=0)
        pivot_lang_abv, target_lang_abv = file.split('/')[-2].split('-')[0], file.split('/')[-2].split('-')[1]
        target_lang = [v for k, v in languages.items() if k == target_lang_abv][0]
        pivot_lang = 'English' if pivot_lang_abv == 'en' else 'French'

        print(f'language: {target_lang}')
        responses = []
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

            input_mes = message

            result = prompt_llm(model_pipeline, tokenizer, input_mes)
            responses.append(result)

            if index % 100 == 0:
                print(index, 'processed', result)


        completions = []
        for completion_text in responses:
            completions.append(completion_text)
        df['mt0'] = completions
        if reverse:
            df.to_csv(output_dir + f'{target_lang_abv}-{pivot_lang_abv}' + '.tsv', sep='\t')
        else:
            df.to_csv(output_dir + f'{pivot_lang_abv}-{target_lang_abv}' + '.tsv', sep='\t')


def machine_translation_germany(model_pipeline, tokenizer, output_dir, reverse=False):
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

        responses = []
        for index in range(df.shape[0]):
            if not reverse:
                text = df['eng_Latn'].iloc[index] if pivot_lang_abv == 'eng' else df['fra_Latn'].iloc[index]
                prompt_query = f"Translate the {pivot_lang} sentence below to {target_lang}. Return the translated sentence only. If you cannot translate the sentence simply say you don't know"
            else:
                text = df['deu_Latn'].iloc[index]
                prompt_query = f"Translate the {target_lang} sentence below to {pivot_lang}. Return the translated sentence only. If you cannot translate the sentence simply say you don't know"

            message = prompt_query + '\n' + text

            input_mes =  message

            result = prompt_llm(model_pipeline, tokenizer, input_mes)
            responses.append(result)

            if index % 100 == 0:
                print(index, 'processed', result)


        completions = []
        for completion_text in responses:
            completions.append(completion_text)
        df['mt0'] = completions
        if reverse:
            df.to_csv(output_dir + f'{target_lang_abv}-{pivot_lang_abv}' + '.tsv', sep='\t')
        else:
            df.to_csv(output_dir + f'{pivot_lang_abv}-{target_lang_abv}' + '.tsv', sep='\t')


def named_entity_recognition(model_pipeline, tokenizer, output_dir):
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

        responses = []
        for index in range(df.shape[0]):
            text = df['text'].iloc[index]
            message = text + '\n\n' + prompt_query
            input_mes = message

            result = prompt_llm(model_pipeline, tokenizer, input_mes)
            responses.append(result)

            if index % 100 == 0:
                print(index, 'processed', result)

        completions = []
        for completion_text in responses:
            completions.append(completion_text)
        df['mt0'] = completions
        file_lang = file.split('/')[-1].split('.')[0]
        df.to_csv(output_dir + file_lang + '.tsv', sep='\t')
        print(file_lang)
        print(completions[:3])



def main(senti: bool = False,
         news: bool = False,
         qa: bool = False,
         qah: bool = False,
         mt_from_en: bool = False,
         mt_to_en: bool = False,
         mtg_from_g: bool = False,
         mtg_to_g: bool = False,
         ner: bool = False,
         summ: bool = False):
    """ Runs the task functions"""

    model_name = 'meta-llama/Llama-2-7b-chat-hf'
    model_pipeline, tokenizer = load_llama(model_name)


    if senti is True:
        output_dir = 'results_mt0_xxl/sentiment/'
        create_dir(output_dir)

        sentiment(model_pipeline, tokenizer, output_dir)
    elif news is True:
        output_dir = 'results_mt0_xxl/news_topic/'
        create_dir(output_dir)

        news_classification(model_pipeline, tokenizer, output_dir)
    elif qa is True:
        output_dir = 'results_mt0_xxl/qa/'
        create_dir(output_dir)

        cross_lingual_qa(model_pipeline, tokenizer, output_dir, pivot=True)
    elif qah is True:
        output_dir = 'results_mt0_xxl/qah/'
        create_dir(output_dir)

        cross_lingual_qa(model_pipeline, tokenizer, output_dir, pivot=False)
    elif mt_to_en is True:

        output_dir = 'results_mt0_xxl/mt/'
        create_dir(output_dir)

        machine_translation(model_pipeline, tokenizer, output_dir, reverse=True)
    elif mt_from_en is True:

        output_dir = 'results_mt0_xxl/mt/'
        create_dir(output_dir)

        machine_translation(model_pipeline, tokenizer, output_dir, reverse=False)
    elif mtg_from_g is True:

        output_dir = 'results_mt0_xxl/mt_german_false/'
        create_dir(output_dir)

        machine_translation_germany(model_pipeline, tokenizer, output_dir, reverse=False)
    elif mtg_to_g is True:

        output_dir = 'results_mt0_xxl/mt_german/'
        create_dir(output_dir)

        machine_translation_germany(model_pipeline, tokenizer, output_dir, reverse=True)
    elif ner is True:

        output_dir = 'results_mt0_xxl/ner/'
        create_dir(output_dir)

        named_entity_recognition(model_pipeline, tokenizer, output_dir)


if __name__ == '__main__':
    main(senti=True)
    main(news=True)
    main(qa=True)
    main(qah=True)
    main(mt_to_en=True)
    main(mt_from_en=True)
    main(ner=True)
    main(mtg_to_g=True)
    main(mtg_from_g=True)
