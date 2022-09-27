import json
from tkinter import dialog
from data_api import HybridDialogueDataset, get_hash
import os
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


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


def passage_ranking(query, dataset):
    files = list(os.listdir(dataset.orig_data_dir))
    files_prepped = list(map(lambda x: prep_file_names(x), files))
    tokenized_corpus = [doc.split(" ") for doc in files_prepped]
    bm25 = BM25Okapi(tokenized_corpus)

    query = query.lower()
    tokenized_query = query.split(" ")
    # doc_scores = bm25.get_scores(tokenized_query)
    top_result = ' '.join(bm25.get_top_n(tokenized_query, tokenized_corpus, n=1)[0])
    return top_result


def get_rank(query, correct_source, num_retrieve=10):
    files = list(os.listdir(dataset.orig_data_dir))
    files_prepped = list(map(lambda x: prep_file_names(x), files))
    tokenized_corpus = [doc.split(" ") for doc in files_prepped]
    bm25 = BM25Okapi(tokenized_corpus)

    query = query.lower()
    tokenized_query = query.split(" ")
    results = bm25.get_top_n(tokenized_query, tokenized_corpus, n=num_retrieve)
    results = list(map(lambda x: ' '.join(x), results))

    if correct_source not in results:
        return num_retrieve + 1
    else:
        return results.index(correct_source) + 1


def error_analysis(turn_id, dataset, num_retrieve=10):
    turn = dataset.get_turn(turn_id)
    query = turn['current_query']
    correct_candidate = candidates[turn['correct_next_cands_ids'][0]]

    correct_source = correct_candidate['page_key'] or correct_candidate['table_key'].rsplit('_', 1)[0]

    correct_source = correct_source.replace("_", ' ').lower()
    # correct_source = correct_source.lower()
    files = list(os.listdir(dataset.orig_data_dir))
    files_prepped = list(map(lambda x: prep_file_names(x), files))
    tokenized_corpus = [doc.split(" ") for doc in files_prepped]
    bm25 = BM25Okapi(tokenized_corpus)

    query = query.lower()
    tokenized_query = query.split(" ")
    results = bm25.get_top_n(tokenized_query, tokenized_corpus, n=num_retrieve)
    results = list(map(lambda x: ' '.join(x), results))

    print(f"query: {query}")
    print(f"correct source: {correct_source}")
    print(f"results:{results}") 


def answer_retrieval():
    model = SentenceTransformer('all-MiniLM-L6-v2')


    
def process_data(dataset):
    conversations = dataset.get_conversations(mode='train')
    candidates = dataset.get_all_candidates()
    turn_ids = dataset.get_turn_ids(mode="train")
    turns = dataset.get_turns(mode="train")

    for key, turn_keys in conversations.items():
        dialogue_history = ''
        data_points = []
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
                if correct_reference in incorrect_references:
                    incorrect_references = turn['possible_next_cands_ids'].remove(correct_reference)

                for incorrect_reference in incorrect_references:
                    incorrect_reference_linearized = candidates[incorrect_reference]['linearized_input']
                    data_point = [H, correct_reference_linearized, incorrect_reference_linearized]
                    data_points.append(data_point)

                response = turn['long_response_to_query']
                dialogue_history += response + ' ' 

        break



        


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



    




