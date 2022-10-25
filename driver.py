import utils
from data_api import HybridDialogueDataset, get_hash
import pickle
from load_data import HybridDialogue_Triplets
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    # Create the triplet samples csv file
    # dataset = HybridDialogueDataset()
    # data_points = utils.create_triplet_samples(dataset, mode='test')
    # data_points.to_csv('triplet_samples_test.csv', index=False)

    # Load the triplet samples 
    # triplet_data = HybridDialogue_Triplets(data_file='triplet_samples.csv')
    # triplet_dataloader = DataLoader(triplet_data, batch_size=32, shuffle=True)

    # for i,batch in enumerate(triplet_dataloader):
    #    history, correct, incorrect = batch
    #    break

    dataset = HybridDialogueDataset()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    data_points = utils.create_dialogue_turns(dataset, tokenizer, mode='train')
    data_points.to_csv('dialogue_samples.csv', index=False)