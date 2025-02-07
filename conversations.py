import utils, metrics 
from data_api import HybridDialogueDataset, get_hash
from transformers import RobertaTokenizer, RobertaModel
import tqdm, pickle
import pandas as pd
import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader 
import torch
import torch.nn as nn 
from torch.optim import Adam, AdamW
from transformers import get_scheduler
from loss import NTXentLoss, CustomNegLoss
import pdb
from torch.nn.utils.rnn import pad_sequence
import gc
from models import PQCLR, PQNCLR

utils.seed_everything(seed=42)


def custom_collate_fn(batch):
    # pdb.set_trace()
    pos = torch.tensor([batch[i][0] for i in range(len(batch))])
    anch = torch.tensor([batch[i][1] for i in range(len(batch))])
    neg = torch.tensor([batch[i][2] for i in range(len(batch))])
    padded = pad_sequence(neg, batch_first=True)
    return (pos, anch, padded)
    

class Passage_Positive_Anchors_Negatives_Dataset(Dataset):
    def __init__(self, positive_encodings, anchor_encodings, negative_encodings):
        self.positive_encodings = positive_encodings
        self.anchor_encodings = anchor_encodings
        self.negative_encodings = negative_encodings

    def __len__(self):
        return len(self.positive_encodings)
    
    def __getitem__(self, idx):
        # positive = self.positives[idx]
        # anchor = self.anchors[idx]
        # examples = InputExample(texts=[positive,anchor])
        # negs = [self.negative_encodings[i][idx] for i in range(len(self.negative_encodings))]
        return self.positive_encodings[idx].clone().detach(), self.anchor_encodings[idx].clone().detach(), self.negative_encodings[:,idx,:].clone().detach()


precomputed = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
# Compute Embeddings using Roberta
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# num_added_toks = tokenizer.add_tokens(['[PARAGRAPH]', '[CELL]', '[ROW]', '[TABLE]'])

# passage_model = RobertaModel.from_pretrained('roberta-base').to(device)
# question_model = RobertaModel.from_pretrained('roberta-base').to(device)
# passage_model.resize_token_embeddings(len(tokenizer))


dataset = HybridDialogueDataset()

# conversations = dataset.get_conversations(mode='train')
# candidates = dataset.get_all_candidates()

# turn_ids = dataset.get_turn_ids(mode="train")
# turns = dataset.get_turns(mode="train")

# dataset.ott_data_dir = '../OTT-QA/data/combined_jsons/'
# dataset.orig_data_dir = '../OTT-QA/data/traindev_tables_tok/'
# dataset.orig_wiki_data_dir = '../OTT-QA/data/traindev_request_tok/'

# evaluated_conversations = []
# evaluated_val_conversations = []

# positives = []
# negatives = []
# anchors = []

data_points = pd.read_csv('triplet_samples_train_new.csv')
data_points = data_points.groupby(['history','correct_reference'])['incorrect_reference'].apply(list).reset_index(name='incorrect_reference')

H = list(data_points['history'])
C = list(data_points['correct_reference'])
# I = list(data_points['incorrect_reference'])

# Trying to make all the negatives the same length for efficient batching later on
len_negatives = 31

# Load the negatives precomputed by utils.generate_negative_samples()
I_padded = []
for i in range(4): # Split the file into 4 parts
    with open(f'negative_samples_part_{i}.pickle', 'rb') as f:
        temp = pickle.load(f)
    
    I_padded.extend(temp)

neg_df = pd.DataFrame(I_padded, columns=[f'neg_{i:02d}' for i in range(31)])
for col in neg_df.columns:
    data_points[col] = neg_df[col]

data_points = data_points.drop('incorrect_reference', axis=1)

val_data_points = pd.read_csv('triplet_samples_validate_new.csv')
val_data_points = val_data_points.groupby(['history', 'correct_reference'])['incorrect_reference'].apply(list).reset_index(name='incorrect_reference')

H_val = list(val_data_points['history'])
C_val = list(val_data_points['correct_reference'])
# I_val = list(val_data_points['incorrect_reference'])

# Doing the same for validation
I_padded_val = []
for i in range(4):
    with open(f'negative_samples_validate_part_{i}.pickle', 'rb') as f:
        temp = pickle.load(f)
    
    I_padded_val.extend(temp)

neg_df_val = pd.DataFrame(I_padded_val, columns=[f'neg_{i:02d}' for i in range(31)])
for col in neg_df_val.columns:
    val_data_points[col] = neg_df_val[col]

val_data_points = val_data_points.drop('incorrect_reference', axis=1)

# Tokenize the datasets
neg_encodings = []

print("Tokenizing...")

if precomputed:
    pos_encodings = torch.load('positive_encodings.pt')
    anchor_encodings = torch.load('anchor_encodings.pt')
    neg_encodings = torch.load('neg_encodings.pt')
    # with open('negative_encodings.pickle', 'rb') as f:
    #     neg_encodings = pickle.load(f)
else:
    pos_encodings = tokenizer(C, padding="max_length", truncation=True, return_tensors='pt')['input_ids']
    anchor_encodings = tokenizer(H, padding="max_length", truncation=True, return_tensors='pt')['input_ids']
    for column in data_points.columns:
        if 'neg' in column:
            temp = tokenizer(list(data_points[column]), padding="max_length", truncation=True, return_tensors='pt')['input_ids']
            neg_encodings.append(temp)

    neg_encodings = torch.stack(neg_encodings)
    torch.save(pos_encodings, 'positive_encodings.pt')
    torch.save(anchor_encodings, 'anchor_encodings.pt')
    torch.save(neg_encodings, 'neg_encodings.pt')


val_anchor_encodings = tokenizer(H_val, padding="max_length", truncation=True, return_tensors='pt')['input_ids']
val_pos_encodings = tokenizer(C_val, padding="max_length", truncation=True, return_tensors='pt')['input_ids']

val_neg_encodings = []
# for item in I_val:
    # temp = tokenizer(item, padding="max_length", truncation=True, return_tensors='pt')['input_ids']
    # val_neg_encodings.append(temp)
for column in val_data_points.columns:
        if 'neg' in column:
            temp = tokenizer(list(val_data_points[column]), padding="max_length", truncation=True, return_tensors='pt')['input_ids']
            val_neg_encodings.append(temp)

val_neg_encodings = torch.stack(val_neg_encodings)

print("Done Tokenizing")

passage_dataset = Passage_Positive_Anchors_Negatives_Dataset(positive_encodings=pos_encodings, anchor_encodings=anchor_encodings, negative_encodings=neg_encodings)
val_passage_dataset = Passage_Positive_Anchors_Negatives_Dataset(positive_encodings=val_pos_encodings, anchor_encodings=val_anchor_encodings, negative_encodings=val_neg_encodings)
batch_size = 4
num_epochs = 15

train_dataloader = DataLoader(passage_dataset, batch_size=batch_size, shuffle=False)#, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_passage_dataset, batch_size=batch_size, shuffle=False)#, collate_fn=custom_collate_fn)


model = PQNCLR(device=device)
model = model.train()

# if torch.cuda.device_count()>1:
    # model = torch.nn.DataParallel(model)


opt1 = Adam(model.passage_encoder.parameters(), lr=1e-6)
opt2 = Adam(model.question_encoder.parameters(), lr=1e-6)

num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler_1 = get_scheduler(name="linear", optimizer=opt1, num_warmup_steps=5, num_training_steps=num_training_steps)
lr_scheduler_2 = get_scheduler(name="linear", optimizer=opt2, num_warmup_steps=5, num_training_steps=num_training_steps)

loss_fn = CustomNegLoss(device=device, batch_size=batch_size, temperature=1, use_cosine_similarity=False, alpha_weight=1)

for epoch in tqdm.trange(num_epochs):
    torch.cuda.empty_cache()
    loss_avg = 0
    num_exp = 0

    top_3_acc = 0
    top_1_acc = 0

    model = model.train()
    for i, batch in enumerate(tqdm.tqdm((train_dataloader))):
        torch.cuda.empty_cache()
        # opt1.zero_grad()
        # opt2.zero_grad()

        pos,anch,neg = batch[0].clone().detach().to(device), batch[1].clone().detach().to(device),  batch[2].clone().detach().to(device)
        # if neg.shape[1] > 43:
        #     neg = neg[:,:43,:]
        
        pos_emb, anch_emb, neg_emb = model(pos, anch, neg)
        # pdb.set_trace()
        loss_val = loss_fn(pos_emb, anch_emb, neg_emb)
        loss_val.backward()

        torch.cuda.empty_cache()

        loss_avg += loss_val.item()*len(pos_emb)
        num_exp += len(pos_emb)

        # if i % 100 == 0:

        opt1.step()
        opt2.step()
        lr_scheduler_1.step()
        lr_scheduler_2.step()
        # break
        # print(loss_val.item())
        del pos, anch, neg, pos_emb
    loss_avg /= num_exp
    print(loss_avg)

    model = model.eval()
    top_3_incorrects = []
    for i, batch in tqdm.tqdm(enumerate(val_dataloader), desc="Validating"):
        pos_val, anch_val, neg_val = batch[0].clone().detach().to(device), batch[1].clone().detach().to(device), batch[2].clone().detach().to(device)
        with torch.no_grad():
            pos_emb_val, anch_emb_val, neg_emb_val = model(pos_val, anch_val, neg_val)
        
        # distances = torch.matmul(pos_emb, torch.transpose(anch_emb, 0, 1))

        correct_dist = torch.dot(pos_emb_val[0,:], anch_emb_val[0,:]).item()
        neg_dists = [torch.dot(anch_emb_val[0,:], neg_emb_val[i,0,:]).item() for i in range(neg_emb_val.shape[0])]

        less_eq = list(map(lambda x: x<correct_dist, neg_dists))
        if sum(less_eq)==0:
            top_1_acc += 1
        
        if sum(less_eq) <3:
            top_3_acc += 1

        # top_1_preds = torch.argmax(distances, dim=0)
        # top_1_acc = torch.sum(gold_answers==top_1_preds)
        
        # for i in range(distances.shape[0]):
        #    if i in torch.argsort(distances[:,i])[-3:]:
        #        top_3_acc += 1
        #    else:
        #        top_3_incorrects.append(i)

        #accuracy = top_1_acc.item()/len(top_1_preds)
        #val_loss = loss_fn(pos_emb, anch_emb)
        # print(f"{val_loss.item()}")
        print(f"top 1 accuracy {top_1_acc}")
        print(f"top 3 accuracy {top_3_acc}")
