import torch
from torch.utils.data import Dataset

class PreferenceDataset(Dataset):
    """
    Dataset for human preference data.
    """
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
    
    def load_data(self, data_path):
        # Load preference data from file
        with open(data_path, 'r') as f:
            data = [line.strip().split('\t') for line in f]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids_1 = torch.tensor(self.data[idx][0])
        input_ids_2 = torch.tensor(self.data[idx][1]) 
        reward = torch.tensor(float(self.data[idx][2]))
        return input_ids_1, input_ids_2, reward 