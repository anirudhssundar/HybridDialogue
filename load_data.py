import json
from data_api import HybridDialogueDataset, get_hash
import os


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




