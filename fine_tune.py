from transformers import AdamW
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers import evaluation
# from transformers import BertTokenizer

# from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from load_data import HybridDialogue_Triplets
from torch.utils.data import Dataset, DataLoader
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Set up training data
training_data = HybridDialogue_Triplets('triplet_samples.csv')
train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
train_loss = losses.TripletLoss(model)

# Set up validation data 
validation_data = pd.read_csv('triplet_samples_val.csv')
history = list(validation_data['history'])
positives = list(validation_data['correct_reference'])
negatives = list(validation_data['incorrect_reference'])

evaluator = evaluation.TripletEvaluator(anchors=history, positives=positives, negatives=negatives, main_distance_function=0, batch_size=64, show_progress_bar=True, name='FROM_SCRATCH', write_csv=True)

# Fit HybriDialogue data to pre-trained SentenceBERT
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, output_path='results/', warmup_steps=100, evaluator=evaluator, evaluation_steps=500, checkpoint_path='checkpoints/', checkpoint_save_steps=500, checkpoint_save_total_limit=5)



