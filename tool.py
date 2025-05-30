import random
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
#Setting Random Seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TPDataset(Dataset):
    def __init__(self, data, labels, species):
        self.data = data
        self.labels = labels
        self.species = species

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.species[idx]

#Calculate cross-entropy loss
def cross_entropy_loss_with_soft_target(preds, soft_targets):
    log_softmax_preds = torch.log_softmax(preds, dim=1)
    loss = -(soft_targets * log_softmax_preds).sum(dim=1)
    return loss.mean()

def species_to_one_hot(number, length=5):
    return [[1 if j == i else 0 for j in range(length)]for i in number]

def ISM_Feature(number_sequences):
    config_path = "../ISM/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(config_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config_path)
    out_data = []
    for i in range(len(number_sequences)):
        input_text = number_sequences[i]
        if len(input_text) > 200:
            input_text = input_text[:200]
        if len(input_text) < 200:
            for _ in range(200 - len(input_text)):
                input_text = input_text + '<pad>'
        batch_labels = tokenizer(input_text, return_tensors="pt").to(device)
        out = model(batch_labels['input_ids'].to("cuda"),attention_mask=batch_labels['attention_mask'].to(device))
        if device  == "cuda":
            out_data.append(out.last_hidden_state.detach().cpu().numpy().squeeze(0))
        else:
            out_data.append(out.last_hidden_state.detach().numpy().squeeze(0))
    return out_data