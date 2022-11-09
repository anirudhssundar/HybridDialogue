from data_api import HybridDialogueDataset, get_hash
import json
import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch 
import re
import os

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


# Creates data sample triplets 
def create_triplet_samples(dataset, mode='train'):
    conversations = dataset.get_conversations(mode=mode)
    candidates = dataset.get_all_candidates()
    turn_ids = dataset.get_turn_ids(mode=mode)
    turns = dataset.get_turns(mode=mode)

    # data_points = []
    data_points = pd.DataFrame()

    history = [] # dialogue history
    correct_reference_all = [] # Correct references
    incorrect_reference_all = [] # Incorrect references

    for key, turn_keys in tqdm.tqdm(conversations.items()):
        dialogue_history = ''
        for i,turn_key in enumerate(turn_keys):
            if i == 0:
                turn = turns[turn_key]
                query = turn['current_query']
                response = turn['long_response_to_query']
                dialogue_history += query + ' '
                dialogue_history += response + ' ' 
                continue  # No need to pick the right reference 

            else: 
                turn = turns[turn_key]
                query = turn['current_query']

                H = dialogue_history + query + ' ' 
                correct_reference = turn['correct_next_cands_ids'][0]
                correct_reference_linearized = candidates[correct_reference]['linearized_input']
                
                incorrect_references = turn['possible_next_cands_ids']
                # print(incorrect_references)
                if correct_reference in incorrect_references:
                    incorrect_references.remove(correct_reference)

                for incorrect_reference in incorrect_references:
                    incorrect_reference_linearized = candidates[incorrect_reference]['linearized_input']
                    # data_point = [H, correct_reference_linearized, incorrect_reference_linearized]
                    # data_points.append(data_point)
                    
                    history.append(H)
                    correct_reference_all.append(correct_reference_linearized)
                    incorrect_reference_all.append(incorrect_reference_linearized)

                response = turn['long_response_to_query']
                dialogue_history = H + response + ' ' 

    data_points['history'] = history
    data_points['correct_reference'] = correct_reference_all
    data_points['incorrect_reference'] = incorrect_reference_all

    return data_points



def create_dialogue_turns(dataset, tokenizer, mode='train'):
    """
    Create the pandas dataframe of contexts (Question, Reference, Answer, Q,R,A...) and responses (Answer)
    """
    eos_token = tokenizer.eos_token

    conversations = dataset.get_conversations(mode=mode)
    candidates = dataset.get_all_candidates()
    turn_ids = dataset.get_turn_ids(mode=mode)
    turns = dataset.get_turns(mode=mode)

    # data_points = []
    data_points = pd.DataFrame()

    history = [] # dialogue history
    answers = []
    correct_reference_all = [] # Correct references
    for key, turn_keys in tqdm.tqdm(conversations.items()):
        dialogue_history = ''
        for i,turn_key in enumerate(turn_keys):
            turn = turns[turn_key]
            query = turn['current_query']
            response = turn['long_response_to_query']
            correct_reference = turn['correct_next_cands_ids'][0]
            correct_reference_linearized = candidates[correct_reference]['linearized_input']

            eos_token = tokenizer.eos_token

            H = f'{dialogue_history} {correct_reference_linearized} {query} {eos_token}' 
            history.append(H)
            answers.append(response)

            dialogue_history = f'{H} {response} {eos_token}'

    data_points['context'] = history
    data_points['response'] = answers

    data_points = data_points.dropna()

    return data_points



class ConversationDataset(Dataset):
    def __init__(self, tokenizer, df, block_size=512):

        # block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        self.examples = []
        for _, row in df.iterrows():
            # conv = construct_conv(row, tokenizer)
            sample = row[0] + row[1] + tokenizer.eos_token
            self.examples.append(sample)

        # logger.info("Saving features into cached file %s", cached_features_file)
        # with open(cached_features_file, "wb") as handle:
        #     pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def find_special_tokens():
    # Find all tokens like [ROW], [CELL], etc in dataset
    training_data = pd.read_csv('triplet_samples.csv')
    positives = list(training_data['correct_reference'])
    set_unique = set()
    for positive in positives:
        unique = re.search('\\[(.*?)\\]', positive).group(0)
        set_unique.add(unique)
    
    print(set_unique)


def create_table_top_level_info(dataset):
    # Go through all tables in the dataset
    # For each table, extract the title, section title, section text, and intro to be encoded
    dataset.orig_data_dir = '../OTT-QA/data/traindev_tables_tok/'

    main_dir = dataset.orig_data_dir

    top_level_info = pd.DataFrame()

    table_titles = []
    table_info = []
    table_ids = []

    for file in os.listdir(main_dir):
        with open(f'{main_dir}/{file}','r') as f:
            data = json.load(f)

        info = ''
        # title = data['title'].lower()
        title = data['uid'].rsplit('_',1)[0].replace('_',' ').lower()
        keys_to_use = ['title','section_title','section_text','intro']
        
        for key in keys_to_use:
            temp = key.replace('_',' ')
            info += f'{temp} is {data[key]}. '

        id = int(data['uid'].rsplit('_',1)[1]) # If there are multiple tables for the same page, need the number of the page too

        table_titles.append(title)
        table_info.append(info)
        table_ids.append(id)

    top_level_info['titles'] = table_titles
    top_level_info['info'] = table_info
    top_level_info['id'] = table_ids 
    return top_level_info


def generate_correct_sources_list(dataset, mode='train'):
    # Generate a list of all possible sources referred to in conversations in the train/[dev]/[test] set
    correct_sources = []
    evaluated_conversations = []
    
    conversations = dataset.get_conversations(mode='train')
    candidates = dataset.get_all_candidates()
    turn_ids = dataset.get_turn_ids(mode="train")
    turns = dataset.get_turns(mode="train")

    for turn_id in tqdm.tqdm(turn_ids):
        turn = dataset.get_turn(turn_id)
        if turn['conversation_id'] in evaluated_conversations:
            continue # Only looking at the first turns 
    
        evaluated_conversations.append(turn['conversation_id'])

        correct_candidate = candidates[turn['correct_next_cands_ids'][0]]

        correct_source = correct_candidate['page_key'] or correct_candidate['table_key'].rsplit('_', 1)[0]
        correct_source = correct_source.replace("_", ' ').lower()

        correct_sources.append(correct_source)
    
    return list(set(correct_sources))


