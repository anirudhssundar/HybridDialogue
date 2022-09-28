from data_api import HybridDialogueDataset, get_hash
import json
import tqdm
import pandas as pd

def create_triplet_samples(dataset):
    conversations = dataset.get_conversations(mode='train')
    candidates = dataset.get_all_candidates()
    turn_ids = dataset.get_turn_ids(mode="train")
    turns = dataset.get_turns(mode="train")

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



