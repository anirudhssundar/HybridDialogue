from data_api import HybridDialogueDataset, get_hash
import json
import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch 
import re, os, pickle
import numpy as np
import random
import pdb

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


def seed_everything(seed=4):  # chosen by fair dice roll. guaranteed to be random. https://xkcd.com/221/

    random.seed(seed)
    tseed = random.randint(1, 1e6)
    tcseed = random.randint(1, 1e6)
    npseed = random.randint(1, 1e6)
    ospyseed = random.randint(1, 1e6)
    torch.manual_seed(tseed)
    torch.cuda.manual_seed_all(tcseed)
    np.random.seed(npseed)
    os.environ['PYTHONHASHSEED'] = str(ospyseed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



# Create triplet samples where negatives are all a list, instead of separating them out as in create_triplet_samples()
# This doesn't exactly work, so I'm creating a dataframe by repeating the incorrect responses 
# and then merging them into a list later on after I load the dataframe
def create_triplet_samples_neg_combined(dataset, mode='train'):
    conversations = dataset.get_conversations(mode=mode)
    candidates = dataset.get_all_candidates()
    turn_ids = dataset.get_turn_ids(mode=mode)
    turns = dataset.get_turns(mode=mode)

    # data_points = []
    data_points = pd.DataFrame()

    wiki_dir = dataset.orig_wiki_data_dir
    orig_dir = dataset.orig_data_dir

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
                correct_candidate = candidates[correct_reference]
                correct_reference_linearized = candidates[correct_reference]['linearized_input']

                if correct_candidate['the_type'] == 'row': # Check for linked Wikipedia information
                    table_key = correct_candidate['table_key']
                    page_key = correct_candidate['page_key']

                    assert table_key != None, "table_key is None"
                    assert page_key != None, "page_key is None"
                    is_passage, passage_data = dataset.get_page_data(correct_candidate['page_key']) # Check if data is from passage/table
                    if is_passage:
                        row_string = passage_data['passage']
                    else: 
                        _,table_data = dataset.get_table_data(table_key)
                        row_data = table_data[correct_candidate['row']]
                        row_data = [item for item in row_data if item != ''] # Get rid of empty strings
                        row_string = ' '.join(row_data)
                    correct_reference_linearized = correct_reference_linearized + ' ' + row_string

                incorrect_references = turn['possible_next_cands_ids']
                # print(incorrect_references)
                if correct_reference in incorrect_references:
                    incorrect_references.remove(correct_reference)

                incorrect_reference_linearized_all = []

                for incorrect_reference in incorrect_references:
                    incorrect_candidate = candidates[incorrect_reference]
                    # data_point = [H, correct_reference_linearized, incorrect_reference_linearized]
                    # data_points.append(data_point)
                    if incorrect_candidate['the_type'] == 'row': # Check for linked Wikipedia information
                        table_key = incorrect_candidate['table_key']
                        page_key = incorrect_candidate['page_key']
                        assert table_key != None, "table_key is None"
                        assert page_key != None, "page_key is None"
                        is_passage, passage_data = dataset.get_page_data(incorrect_candidate['page_key']) # Check if data is from passage/table
                        if is_passage:
                            incorrect_row_string = passage_data['passage']
                        else: 
                            _,table_data = dataset.get_table_data(table_key)
                            row_data = table_data[incorrect_candidate['row']]
                            row_data = [item for item in row_data if item != ''] # Get rid of empty strings
                            incorrect_row_string = ' '.join(row_data)
                        incorrect_reference_linearized = candidates[incorrect_reference]['linearized_input'] + ' '+  incorrect_row_string
                    else:
                        incorrect_reference_linearized = candidates[incorrect_reference]['linearized_input']
                    incorrect_reference_linearized_all.append(incorrect_reference_linearized)

                history.append(H)
                correct_reference_all.append(correct_reference_linearized)
                incorrect_reference_all.append(incorrect_reference_linearized_all)
                # pdb.set_trace()

                response = turn['long_response_to_query']
                dialogue_history = H + response + ' ' 

    data_points['history'] = history
    data_points['correct_reference'] = correct_reference_all
    data_points['incorrect_reference'] = incorrect_reference_all

    data_points = data_points.explode('incorrect_reference')
    # pdb.set_trace()

    return data_points


def generate_negative_samples():
    # Generates negative samples by sampling from the same table or sampling other positives
    # Doing this so all data samples have the same number of negatives
    data_points = pd.read_csv('triplet_samples_train_new.csv')
    data_points = data_points.groupby(['history','correct_reference'])['incorrect_reference'].apply(list).reset_index(name='incorrect_reference')

    H = list(data_points['history'])
    C = list(data_points['correct_reference'])
    I = list(data_points['incorrect_reference'])


    # Trying to make all the negatives the same length for efficient batching later on
    len_negatives = 31

    num_parts = 4
    for part in range(num_parts):
        start = part*len(I)//4
        end = (part+1)*len(I)//4
        # print(start, end)
        I_sub = I[start:end]

        I_new = [[0]*len_negatives]*len(I_sub)

        for i, item in enumerate(tqdm.tqdm(I_sub)):
            # gc.disable()
            if len(item) >= len_negatives:
                temp_item = item[:len_negatives]
            else: # Need to sample new negatives
                # temp_prob_dist = copy.deepcopy(prob_dist)
                # temp_prob_dist[i] = 0.0
                # new_samples = list(np.random.choice(C, len_negatives - len(item), replace=False, p=prob_dist[i]))
                new_samples = list(np.random.default_rng().choice(C, len_negatives + 1 - len(item), replace=False))
                if C[i+start] in new_samples:
                    new_samples.remove(C[i+start])
                else:
                    _ = new_samples.pop()
                temp_item = item + new_samples
            
            I_new[i] = temp_item
        
        with open(f'negative_samples_part_{part}.pickle', 'wb') as g:
            pickle.dump(I_new, g)

    return 1