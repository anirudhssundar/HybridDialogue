import utils
from data_api import HybridDialogueDataset, get_hash
import pickle


if __name__ == "__main__":
    dataset = HybridDialogueDataset()
    data_points = utils.create_triplet_samples(dataset)
    # with open('triplet_samples.pickle', 'wb') as f:
    #     pickle.dump(data_points, f)

    data_points.to_csv('triplet_samples.csv', index=False)