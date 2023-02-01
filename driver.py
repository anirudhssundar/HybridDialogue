import utils
from data_api import HybridDialogueDataset, get_hash
import pickle, json
from load_data import HybridDialogue_Triplets
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import RobertaTokenizer, RobertaModel

if __name__ == "__main__":
    # Create the triplet samples csv file
    # dataset = HybridDialogueDataset()
    # data_points = utils.create_triplet_samples(dataset, mode='test')
    # data_points.to_csv('triplet_samples_test.csv', index=False)
    
    # data_points = utils.create_triplet_samples_neg_combined(dataset, mode='train')
    # data_points.to_csv('triplet_samples_with_responses_train.csv')

    # Load the triplet samples 
    # triplet_data = HybridDialogue_Triplets(data_file='triplet_samples.csv')
    # triplet_dataloader = DataLoader(triplet_data, batch_size=32, shuffle=True)

    # for i,batch in enumerate(triplet_dataloader):
    #    history, correct, incorrect = batch
    #    break

    # Create the dialogue turns file
    
    # dataset = HybridDialogueDataset()
    # dataset.ott_data_dir = '../OTT-QA/data/combined_jsons/'
    # dataset.orig_data_dir = '../OTT-QA/data/traindev_tables_tok/'
    # dataset.orig_wiki_data_dir = '../OTT-QA/data/traindev_request_tok/'
    # data_points = utils.create_triplet_samples_neg_combined(dataset, mode='test')
    # data_points.to_csv('triplet_samples_with_responses_test.csv', index=False)

    # data_points = utils.create_triplet_samples_neg_combined(dataset, mode='validate')
    # data_points.to_csv('triplet_samples_with_responses_val.csv', index=False)

    # tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    # model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    # data_points = utils.create_dialogue_turns(dataset, tokenizer, mode='train')
    # data_points.to_csv('dialogue_samples.csv', index=False)
    # utils.generate_negative_samples(mode='validate')
    # dialogue_data_frame = utils.generate_dialogue_response_data(mode='test')

    """
    dialogue_data_frame = utils.generate_dialogue_response_data_all_knowledge(mode='test')
    with open('GODEL_data_test_all_knowledge.json', 'w') as f:
        for idx, row in dialogue_data_frame.iterrows():
            if idx !=0:
                # f.write(',\n')
                f.write('\n')
            # f.write(json.dumps(row.to_dict(), indent=4))
            f.write(json.dumps(row.to_dict()))
    """
    
    model_path = 'fine-tuned-qa_retriever_distributed_epoch_13_Jan_10.pt'
    mode='validate'
    dialogue_data_frame = utils.generate_dialogue_response_top_1s(model_path=model_path, mode=mode)
    with open(f'GODEL_data_{mode}_top_1_knowledge.json', 'w') as f:
        for idx, row in dialogue_data_frame.iterrows():
            if idx !=0:
                # f.write(',\n')
                f.write('\n')
            # f.write(json.dumps(row.to_dict(), indent=4))
            f.write(json.dumps(row.to_dict()))