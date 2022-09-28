import json
from data_api import HybridDialogueDataset, get_hash
import os
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset
import pandas as pd


def rename_jsons_page_key(dataset):
    for file in os.listdir(dataset.orig_data_dir):
        table_key = file.split('.json', 1)[0]
        arr = table_key.rsplit('_', 1)
        page_key, table_num = arr[0], arr[1]
        page_hash = get_hash(page_key)

        with open(dataset.orig_data_dir+file) as f:
            page_data = json.load(f)
        
        hashed_file = dataset.ott_data_dir + page_hash + '.json'
        if os.path.isfile(hashed_file): # already exists
            with open(hashed_file, 'r') as f:
                curr_data = json.load(f)

            curr_data.append(page_data)
            write_object = json.dumps(curr_data)

        else:
            write_object = json.dumps([page_data])
        
        with open(hashed_file,'w') as f:
            f.write(write_object)

        # break


def create_passage_jsons(dataset):
    passage_folder = dataset.orig_wiki_data_dir
    
    for file in os.listdir(passage_folder):
        table_key = file.split('.json', 1)[0]
        arr = table_key.rsplit('_', 1)
        page_key, table_num = arr[0], arr[1]
        page_hash = get_hash(page_key)

        with open(passage_folder+file) as f:
            passage_data = json.load(f)

        for key,value in passage_data.items():
            new_key = key[6:]
            page_hash = get_hash(new_key)
            new_dict = {"passage": value}

            hashed_file = dataset.ott_data_dir + page_hash + '.json'

            write_object = json.dumps(new_dict)
            # "passage"
            with open(hashed_file, 'w') as f:
                f.write(write_object)


def prep_file_names(file_name):
    split_name = file_name.split('.json', 1)[0]
    page_name = split_name.rsplit('_', 1)[0]
    page_name = page_name.replace("_", ' ')
    page_name = page_name.lower()
    return page_name


class HybridDialogue_DataLoader(Dataset):
    def __init__(self, data_file, transform=None):
        self.data = pd.read_csv(data_file)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        history = self.data.iloc[idx, 0]
        correct = self.data.iloc[idx, 1]
        incorrect = self.data.iloc[idx, 2]

        return history, correct, incorrect






if __name__ == "__main__":
    dataset = HybridDialogueDataset()

    conversations = dataset.get_conversations(mode='train')
    candidates = dataset.get_all_candidates()


    turn_ids = dataset.get_turn_ids(mode="train")
    turns = dataset.get_turns(mode="train")

    # Getting a turn:
    turn0 = dataset.get_turn(turn_ids[0])

    dataset.ott_data_dir = '../OTT-QA/data/combined_jsons/'
    dataset.orig_data_dir = '../OTT-QA/data/traindev_tables_tok/'
    dataset.orig_wiki_data_dir = '../OTT-QA/data/traindev_request_tok/'

    # rename_jsons_page_key(dataset)
    # create_passage_jsons(dataset)

    # Evaluate retriever
    acc = 0
    evaluated_conversations = []

    error_turns = []
    MRR = 0

    for turn_id in turn_ids:
        turn = dataset.get_turn(turn_id)
        if turn['conversation_id'] in evaluated_conversations:
            continue
        
        evaluated_conversations.append(turn['conversation_id'])
        query = turn['current_query']

        correct_candidate = candidates[turn['correct_next_cands_ids'][0]]

        correct_source = correct_candidate['page_key'] or correct_candidate['table_key'].rsplit('_', 1)[0]

        correct_source = correct_source.replace("_", ' ').lower()

        # Calculate top-1 accuracy 
        # retrieved_source = passage_ranking(query, dataset)
        # if retrieved_source == correct_source:
        #     acc += 1
        
        # Error analysis 
        # if retrieved_source != correct_source:
        #     error_turns.append(turn_id)

        # Mean reciprocal rank

        rank_source = get_rank(query, correct_source, num_retrieve=10)
        MRR += 1/(rank_source)



    




