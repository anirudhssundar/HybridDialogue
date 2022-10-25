from data_api import HybridDialogueDataset, get_hash
import json
import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

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

    return data_points



class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, block_size=512):

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory = args.cache_dir
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size)
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            for _, row in df.iterrows():
                conv = construct_conv(row, tokenizer)
                self.examples.append(conv)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)




