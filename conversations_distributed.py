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
from models import PQCLR, PQNCLR, PQNTriplet, PQNTriplet_Distributed

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

utils.seed_everything(seed=42)

def prepare(rank, world_size, batch_size=8, pin_memory=False, num_workers=0):
    # dataset = Passage_Triplet_Dataset()
    pos_encodings = torch.load('positive_triplet_encodings.pt')
    anchor_encodings = torch.load('anchor_triple_encodings.pt')
    neg_encodings = torch.load('neg_triplet_encodings.pt')

    passage_dataset = Passage_Triplet_Dataset(positive_encodings=pos_encodings, anchor_encodings=anchor_encodings, negative_encodings=neg_encodings)
    sampler = DistributedSampler(passage_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(passage_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader
    

class Passage_Triplet_Dataset(Dataset):
    def __init__(self, positive_encodings, anchor_encodings, negative_encodings):
        self.positive_encodings = positive_encodings
        self.anchor_encodings = anchor_encodings
        self.negative_encodings = negative_encodings

    def __len__(self):
        return len(self.positive_encodings)
    
    def __getitem__(self, idx):
        return self.positive_encodings[idx].clone().detach(), self.anchor_encodings[idx].clone().detach(), self.negative_encodings[idx].clone().detach()


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


precomputed = True
max_len_tokenizer = 512
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
# Compute Embeddings using Roberta
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer.model_max_length = max_len_tokenizer 
# num_added_toks = tokenizer.add_tokens(['[PARAGRAPH]', '[CELL]', '[ROW]', '[TABLE]'])

# passage_model = RobertaModel.from_pretrained('roberta-base').to(device)
# question_model = RobertaModel.from_pretrained('roberta-base').to(device)
# passage_model.resize_token_embeddings(len(tokenizer))


# dataset = HybridDialogueDataset()

# data_points = pd.read_csv('triplet_samples_train_new.csv')

# H = list(data_points['history'])
# C = list(data_points['correct_reference'])
# I = list(data_points['incorrect_reference'])



# Tokenize the datasets
# neg_encodings = []

# print("Tokenizing...")

"""
if precomputed:
    pos_encodings = torch.load('positive_triplet_encodings.pt')
    anchor_encodings = torch.load('anchor_triple_encodings.pt')
    neg_encodings = torch.load('neg_triplet_encodings.pt')
    # with open('negative_encodings.pickle', 'rb') as f:
    #     neg_encodings = pickle.load(f)
else:
    pos_encodings = tokenizer(C, padding="max_length", truncation=True, max_length=max_len_tokenizer, return_tensors='pt')['input_ids']
    anchor_encodings = tokenizer(H, padding="max_length", truncation=True, max_length=max_len_tokenizer, return_tensors='pt')['input_ids']
    neg_encodings = tokenizer(I, padding="max_length", truncation=True, max_length=max_len_tokenizer, return_tensors='pt')['input_ids']

    torch.save(pos_encodings, 'positive_triplet_encodings.pt')
    torch.save(anchor_encodings, 'anchor_triple_encodings.pt')
    torch.save(neg_encodings, 'neg_triplet_encodings.pt')
"""

# val_anchor_encodings = tokenizer(H_val, padding="max_length", truncation=True, max_length=max_len_tokenizer, return_tensors='pt')['input_ids']
# val_pos_encodings = tokenizer(C_val, padding="max_length", truncation=True, max_length=max_len_tokenizer, return_tensors='pt')['input_ids']
# val_neg_encodings = tokenizer(I_val, padding="max_length", truncation=True, max_length=max_len_tokenizer, return_tensors='pt')['input_ids']

# print("Done Tokenizing")

# passage_dataset = Passage_Triplet_Dataset(positive_encodings=pos_encodings, anchor_encodings=anchor_encodings, negative_encodings=neg_encodings)
# val_passage_dataset = Passage_Triplet_Dataset(positive_encodings=val_pos_encodings, anchor_encodings=val_anchor_encodings, negative_encodings=val_neg_encodings)

batch_size = 8
num_epochs = 5

# train_dataloader = DataLoader(passage_dataset, batch_size=batch_size, shuffle=True)#, collate_fn=custom_collate_fn)
# val_dataloader = DataLoader(val_passage_dataset, batch_size=len(val_passage_dataset), shuffle=False)#, collate_fn=custom_collate_fn)

# model = PQNCLR(device=device)
# model = PQNTriplet(device=device)
# model = model.train()

# opt1 = Adam(model.passage_encoder.parameters(), lr=1e-6)
# opt2 = Adam(model.question_encoder.parameters(), lr=1e-6)

# num_training_steps = num_epochs * len(train_dataloader)

# lr_scheduler_1 = get_scheduler(name="linear", optimizer=opt1, num_warmup_steps=5, num_training_steps=num_training_steps)
# lr_scheduler_2 = get_scheduler(name="linear", optimizer=opt2, num_warmup_steps=5, num_training_steps=num_training_steps)

# loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)


# torch.save(model, 'fine-tuned-qa_retriever_zero_grad_uncommented_lr_1e6_token_1024.pt')

def validate(model_path):
    val_data_points = pd.read_csv('triplet_samples_validate_new.csv')

    print("Tokenizing...")
    H_val = list(val_data_points['history'])
    C_val = list(val_data_points['correct_reference'])
    I_val = list(val_data_points['incorrect_reference'])

    val_anchor_encodings = tokenizer(H_val, padding="max_length", truncation=True, max_length=max_len_tokenizer, return_tensors='pt')['input_ids']
    val_pos_encodings = tokenizer(C_val, padding="max_length", truncation=True, max_length=max_len_tokenizer, return_tensors='pt')['input_ids']
    val_neg_encodings = tokenizer(I_val, padding="max_length", truncation=True, max_length=max_len_tokenizer, return_tensors='pt')['input_ids']

    print("Done tokenizing")

    val_passage_dataset = Passage_Triplet_Dataset(positive_encodings=val_pos_encodings, anchor_encodings=val_anchor_encodings, negative_encodings=val_neg_encodings)
    val_dataloader = DataLoader(val_passage_dataset, batch_size=batch_size, shuffle=False)#, collate_fn=custom_collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PQNTriplet_Distributed(device=device)
    model.load_state_dict(torch.load(f'{model_path}'))
    
    num_less = 0
    total_num = 0
    model = model.eval()

    for i, batch in tqdm.tqdm(enumerate(val_dataloader), desc="Validating"):
        pos_val, anch_val, neg_val = batch[0].clone().detach().to(device), batch[1].clone().detach().to(device), batch[2].clone().detach().to(device)
        with torch.no_grad():
            pos_emb_val, anch_emb_val, neg_emb_val = model(pos_val, anch_val, neg_val)

        pos_emb_val = torch.nn.functional.normalize(pos_emb_val, dim=1)
        anch_emb_val = torch.nn.functional.normalize(anch_emb_val, dim=1)
        neg_emb_val = torch.nn.functional.normalize(neg_emb_val, dim=1)

        pos_dist = torch.bmm(pos_emb_val.view(pos_emb_val.shape[0], 1, pos_emb_val.shape[1]), anch_emb_val.view(anch_emb_val.shape[0],anch_emb_val.shape[1], 1))
        neg_dist = torch.bmm(neg_emb_val.view(neg_emb_val.shape[0], 1, neg_emb_val.shape[1]), anch_emb_val.view(anch_emb_val.shape[0],anch_emb_val.shape[1], 1))

        num_less += sum(pos_dist<neg_dist).item()
        total_num += pos_dist.shape[0]

    print("Val acc is; ", num_less/total_num)





def main(rank, world_size):
    setup(rank, world_size)
    dataloader = prepare(rank, world_size, batch_size=batch_size)
    model = PQNTriplet_Distributed(device=rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    opt1 = Adam(model.module.passage_encoder.parameters(), lr=1e-6)
    opt2 = Adam(model.module.question_encoder.parameters(), lr=1e-6)

    num_training_steps = num_epochs * len(dataloader)

    lr_scheduler_1 = get_scheduler(name="linear", optimizer=opt1, num_warmup_steps=5, num_training_steps=num_training_steps)
    lr_scheduler_2 = get_scheduler(name="linear", optimizer=opt2, num_warmup_steps=5, num_training_steps=num_training_steps)

    loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    device = rank
    for epoch in tqdm.trange(num_epochs):
        dataloader.sampler.set_epoch(epoch)
        torch.cuda.empty_cache()
        loss_avg = 0
        num_exp = 0

        top_3_acc = 0
        top_1_acc = 0

        model = model.train()
        for step, batch in enumerate(tqdm.tqdm((dataloader))):
            opt1.zero_grad(set_to_none=True)
            opt2.zero_grad(set_to_none=True)

            pos,anch,neg = batch[0], batch[1],  batch[2]
            
            pos_emb, anch_emb, neg_emb = model(pos, anch, neg)
            # pdb.set_trace()
            loss_val = loss_fn(anch_emb, pos_emb, neg_emb) # Fixed issue in ordering of elements here 
            loss_val.backward()

            loss_avg += loss_val.item()*len(pos_emb)
            num_exp += len(pos_emb)

            opt1.step()
            opt2.step()
            lr_scheduler_1.step()
            lr_scheduler_2.step()
            del pos, anch, pos_emb
        loss_avg /= num_exp
        print("Epoch: ", loss_avg)

        # model = model.eval()
        # top_3_incorrects = []
        """
        for i, batch in tqdm.tqdm(enumerate(val_dataloader), desc="Validating"):
            pos_val, anch_val = batch[0].clone().detach().to(device), batch[1].clone().detach().to(device)
            with torch.no_grad():
                pos_emb_val, anch_emb_val, neg_emb_val = model(pos_val, anch_val, neg_val)
            
            distances = torch.matmul(pos_emb_val, torch.transpose(anch_emb_val, 0, 1))

            top_1_preds = torch.argmax(distances, dim=0)
            top_1_acc = torch.sum(gold_answers==top_1_preds)
            
            for j in range(distances.shape[0]):
                if j in torch.argsort(distances[:,j])[-3:]:
                    top_3_acc += 1
                else:
                    top_3_incorrects.append(i)

            accuracy = top_1_acc.item()/len(top_1_preds)
            val_loss = loss_fn(pos_emb_val, anch_emb_val)
            print(f"{val_loss.item()}")
            print(f"top 1 accuracy {accuracy}")
            print(f"top 3 accuracy {top_3_acc}")
        """
        # pdb.set_trace()
        torch.distributed.barrier()
        if rank == 0:
            torch.save(model.module.state_dict(), f'fine-tuned-qa_retriever_distributed_epoch_{epoch}_round_2.pt')
        
        validate(f'fine-tuned-qa_retriever_distributed_epoch_{epoch}_round_2.pt')

    cleanup()

    


if __name__ == "__main__":
    world_size = 8
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )