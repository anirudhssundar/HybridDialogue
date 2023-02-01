import utils, metrics 
from data_api import HybridDialogueDataset, get_hash
from transformers import RobertaTokenizer, RobertaModel
import tqdm
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader 
import torch
import torch.nn as nn 
from torch.optim import Adam, AdamW
from transformers import get_scheduler
from loss import NTXentLoss
import pdb
from models import PQCLR, PQNCLR
from rank_bm25 import BM25Okapi

utils.seed_everything(seed=42)

class Passage_Positive_Anchors_Dataset(Dataset):
    def __init__(self, positive_embeddings, anchor_embeddings):
        self.positive_embeddings = positive_embeddings
        self.anchor_embeddings = anchor_embeddings

    def __len__(self):
        return len(self.positive_embeddings)
    
    def __getitem__(self, idx):
        # positive = self.positives[idx]
        # anchor = self.anchors[idx]
        # examples = InputExample(texts=[positive,anchor])
        return self.positive_embeddings[idx].clone().detach(), self.anchor_embeddings[idx].clone().detach()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
# Compute Embeddings using Roberta
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# num_added_toks = tokenizer.add_tokens(['[PARAGRAPH]', '[CELL]', '[ROW]', '[TABLE]'])

# passage_model = RobertaModel.from_pretrained('roberta-base').to(device)
# question_model = RobertaModel.from_pretrained('roberta-base').to(device)
# passage_model.resize_token_embeddings(len(tokenizer))


dataset = HybridDialogueDataset()

conversations = dataset.get_conversations(mode='train')
candidates = dataset.get_all_candidates()

turn_ids = dataset.get_turn_ids(mode="train")
turns = dataset.get_turns(mode="train")

dataset.ott_data_dir = '../OTT-QA/data/combined_jsons/'
dataset.orig_data_dir = '../OTT-QA/data/traindev_tables_tok/'
dataset.orig_wiki_data_dir = '../OTT-QA/data/traindev_request_tok/'

evaluated_conversations = []
evaluated_val_conversations = []
"""
# Let's create a dictionary of candidates and the linearized input required to learn embeddings from 
# Considering only the head of each table, i.e., what is visible in the first turn of conversation 
heads_dict = {}
for key,candidate in candidates.items():
    if candidate['the_type'] not in ['unlinked_paragraph', 'table']:
        continue
    else:
        source = candidate['page_key'] or candidate['table_key'].rsplit('_', 1)[0]
        source = source.replace("_", ' ').lower()
        inpt = candidate['linearized_input']
        combined = f'{source} {inpt}'
        heads_dict[source] = combined
"""


top_level_info_df = utils.create_table_top_level_info(dataset) # pandas df containing tables and their corresponding information

correct_sources = utils.generate_correct_sources_list(dataset, mode='train')
val_correct_sources = utils.generate_correct_sources_list(dataset, mode='validate')

# Generate the contrastive pairs for learning embeddings 

positives = []
negatives = []
anchors = []
sources = []

for turn_id in tqdm.tqdm(turn_ids, desc='Generating pairs'):
    turn = dataset.get_turn(turn_id)
    if turn['conversation_id'] in evaluated_conversations:
        continue # Only looking at the first turns 
    
    evaluated_conversations.append(turn['conversation_id'])
    query = turn['current_query']

    correct_candidate = candidates[turn['correct_next_cands_ids'][0]]

    correct_source = correct_candidate['page_key'] or correct_candidate['table_key'].rsplit('_', 1)[0]
    sources.append(correct_candidate['page_key'] or correct_candidate['table_key'])
    correct_source = correct_source.replace("_", ' ').lower()

    possible_candidates = top_level_info_df[top_level_info_df['titles']==correct_source]
    if len(possible_candidates) > 1: # Multiple tables for this question exist, need to select the correct one
        if correct_candidate['table_key'] is not None:
            correct_id = int(correct_candidate['table_key'].rsplit('_', 1)[1])
            correct_info = list(possible_candidates[possible_candidates['id']==correct_id]['info'])[0]
        else:
            correct_info = possible_candidates.iloc[0]['info']
    else:
        correct_info = possible_candidates.iloc[0]['info']

    positives.append(correct_info)
    anchors.append(query)

    # correct_input = correct_candidate['linearized_input']
    # correct_combined = f'{correct_source} {correct_input}'

    # Calculate top-1 accuracy 
    # retrieved_source = metrics.passage_ranking(query, dataset)

    # Get top BM25 sources
    # retrieved_sources = metrics.return_closest_bm25s(query, dataset, correct_sources, num_retrieve=20, mode="train")

    # incorrect_sources = list(filter(lambda x: x!= correct_source, retrieved_sources))

    # repeated_correct_sources = [correct_source]*len(incorrect_sources) # Repeat to number of negatives
    # positives.extend(repeated_correct_sources)
    # negatives.extend(incorrect_sources)

    # repeated_query = [query]*len(incorrect_sources) # Repeat to number of negatives
    # anchors.extend(repeated_query)

val_conversations = dataset.get_conversations(mode='validate')
val_turn_ids = dataset.get_turn_ids(mode="validate")
val_turns = dataset.get_turns(mode="validate")
val_positives = []
val_anchors = []
val_sources = []

for val_turn_id in tqdm.tqdm(val_turn_ids, desc='Generating pairs'):
    val_turn = dataset.get_turn(val_turn_id)
    if val_turn['conversation_id'] in evaluated_val_conversations:
        continue # Only looking at the first turns 
    
    evaluated_val_conversations.append(val_turn['conversation_id'])
    query = val_turn['current_query']

    correct_candidate = candidates[val_turn['correct_next_cands_ids'][0]]

    correct_source = correct_candidate['page_key'] or correct_candidate['table_key'].rsplit('_', 1)[0]
    val_sources.append(correct_candidate['page_key'] or correct_candidate['table_key'])
    correct_source = correct_source.replace("_", ' ').lower()

    possible_candidates = top_level_info_df[top_level_info_df['titles']==correct_source]
    if len(possible_candidates) > 1: # Multiple tables for this question exist, need to select the correct one
        if correct_candidate['table_key'] is not None:
            correct_id = int(correct_candidate['table_key'].rsplit('_', 1)[1])
            correct_info = list(possible_candidates[possible_candidates['id']==correct_id]['info'])[0]
        else:
            correct_info = possible_candidates.iloc[0]['info']
    else:
        correct_info = possible_candidates.iloc[0]['info']

    val_positives.append(correct_info)
    val_anchors.append(query)

def preprocess_function(examples):
   #function to tokenize the dataset
   return tokenizer(examples, truncation=True, padding=True, return_tensors='pt')['input_ids']


positive_encodings = tokenizer(positives, padding=True, truncation=True, return_tensors='pt')['input_ids']
anchor_encodings = tokenizer(anchors, padding=True, truncation=True, return_tensors='pt')['input_ids']

val_positive_encodings = tokenizer(val_positives, padding=True, truncation=True, return_tensors='pt')['input_ids']
val_anchor_encodings = tokenizer(val_anchors, padding=True, truncation=True, return_tensors='pt')['input_ids']

passage_dataset = Passage_Positive_Anchors_Dataset(positive_embeddings=positive_encodings, anchor_embeddings=anchor_encodings)
val_passage_dataset = Passage_Positive_Anchors_Dataset(positive_embeddings=val_positive_encodings, anchor_embeddings=val_anchor_encodings)

batch_size=16
num_epochs = 20

train_dataloader = DataLoader(passage_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_passage_dataset, batch_size=len(val_passage_dataset), shuffle=False)

model = PQCLR(device=device)
model = model.train()

opt1 = Adam(model.passage_encoder.parameters(), lr=1e-6)
opt2 = Adam(model.question_encoder.parameters(), lr=1e-6)

num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler_1 = get_scheduler(name="linear", optimizer=opt1, num_warmup_steps=5, num_training_steps=num_training_steps)
lr_scheduler_2 = get_scheduler(name="linear", optimizer=opt2, num_warmup_steps=5, num_training_steps=num_training_steps)

loss_fn = NTXentLoss(device=device, batch_size=batch_size, temperature=1, use_cosine_similarity=False, alpha_weight=1)
gold_answers = torch.arange(242).to(device)

# Create a list to keep track of top 3 and extract the 2 closest retrievals to improve downstream eval
val_top_3_list = []

for epoch in tqdm.trange(num_epochs):
    loss_avg = 0
    num_exp = 0

    top_3_acc = 0

    model = model.train()
    for i, batch in enumerate(train_dataloader):
        opt1.zero_grad()
        opt2.zero_grad()
        
        pos,anch = batch[0].clone().detach().to(device), batch[1].clone().detach().to(device)
        pos_emb, anch_emb = model(pos, anch)
        # pdb.set_trace()
        loss_val = loss_fn(pos_emb, anch_emb)
        loss_val.backward()

        loss_avg += loss_val.item()*len(pos_emb)
        num_exp += len(pos_emb)

        opt1.step()
        opt2.step()
        lr_scheduler_1.step()
        lr_scheduler_2.step()
        # break
        # print(loss_val.item())
    loss_avg /= num_exp
    print(loss_avg)

    model = model.eval()
    top_3_incorrects = []
    top_ten_reciprocal_ranks = []
    for i, batch in tqdm.tqdm(enumerate(val_dataloader), desc="Validating"):
        pos, anch = batch[0].clone().detach().to(device), batch[1].clone().detach().to(device)
        with torch.no_grad():
            pos_emb, anch_emb = model(pos, anch)
        
        distances = torch.matmul(pos_emb, torch.transpose(anch_emb, 0, 1))
        top_1_preds = torch.argmax(distances, dim=0)
        top_1_acc = torch.sum(gold_answers==top_1_preds)
        
        for j in range(distances.shape[0]):
            if j in torch.argsort(distances[:,j])[-3:]:
                top_3_acc += 1
            else:
                top_3_incorrects.append(j)
            
            if j in torch.argsort(distances[:,j])[-10:]:
                reversed_positions = torch.flip(torch.argsort(distances[:,j])[-10:],dims=(0,)) # Make it so the top retrieved is first
                position = (reversed_positions==j).nonzero().item() +1
                top_ten_reciprocal_ranks.append(1/position)

                closest_3_negs = reversed_positions[:4].tolist()
                if j in closest_3_negs:
                    closest_3_negs.remove(j)
                else:
                    closest_3_negs = closest_3_negs[:3]
                val_top_3_list.append(closest_3_negs)

            else : 
                top_ten_reciprocal_ranks.append(0)
            

        accuracy = top_1_acc.item()/len(top_1_preds)
        val_loss = loss_fn(pos_emb, anch_emb)
        print(f"{val_loss.item()}")
        print(f"top 1 accuracy {accuracy}")
        print(f"top 3 accuracy {top_3_acc/pos.shape[0]}")

# pdb.set_trace()

torch.save(model, 'fine-tuned-encoder.pt')
with open('closest_3_negs.pickle', 'wb') as f:
    pickle.dump(val_top_3_list, f)




# BM25 evaluation 
"""
corpus = val_positives
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

bm25_top_10s = []

queries = val_anchors
for i,query in enumerate(queries):
    tokenized_query = query.split(" ")
    top_10 = bm25.get_top_n(tokenized_query, corpus, n=10)
    
    if corpus[i] in top_10:
        position = top_10.index(corpus[i])
        bm25_top_10s.append(1/(position+1))
    else:
        bm25_top_10s.append(0)
"""

# BM25 get top 3 from training data

corpus = positives
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

bm25_top_3s = []

queries = anchors
for i,query in tqdm.tqdm(enumerate(queries)):
    tokenized_query = query.split(" ")
    top_3 = bm25.get_top_n(tokenized_query, corpus, n=4)
    
    if corpus[i] in top_3:
        top_3.remove(corpus[i])
    else:
        top_3 = top_3[:3]

    top_3_indices = list(map(lambda x: sources[positives.index(x)], top_3))
    bm25_top_3s.append(top_3_indices)

    # if corpus[i] in top_3:
    #     position = top_3.index(corpus[i])
    #     bm25_top_3s.append(1/(position+1))
    # else:
    #     bm25_top_3s.append(0)

