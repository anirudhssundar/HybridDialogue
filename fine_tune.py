from transformers import AdamW
from sentence_transformers import SentenceTransformer, InputExample, losses
# from transformers import BertTokenizer

# from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from load_data import HybridDialogue_Triplets
from torch.utils.data import Dataset, DataLoader
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
optimizer = torch.optim.Adam(model.parameters())

training_data = HybridDialogue_Triplets('triplet_samples.csv')
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
train_loss = losses.TripletLoss(model)

for i, batch in enumerate(train_dataloader):
    history = batch[0]
    correct_reference = batch[1]
    incorrect_reference = batch[2]
    

    # train_example = [InputExample(texts=[history, correct_reference, incorrect_reference])]
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

sentences = ["[PARAGRAPH] Hero's Hero's was a Japanese mixed martial arts promotion operated by Fighting and Entertainment Group, the parent entity behind kickboxing organization K-1. Grown from and branched off of K-1's earlier experiments in MMA, including the K-1 Romanex event and various MMA fights on its regular K-1 kickboxing cards, it held its first show on March 26, 2005. The promotion was handled by former Rings head Akira Maeda. At a press conference on February 13, 2008,FEG announced that they discontinued Hero's and were creating a new mixed martial arts franchise, Dream, in collaboration with former Pride FC executives from Dream Stage Entertainment.", '[TABLE] Events $ $']

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)


