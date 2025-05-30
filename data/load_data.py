import numpy as np
import pandas as pd
import torch
from DeepMaT.tool import TPDataset

amino_acid_to_int = {'X': 0, 'M': 1, 'I': 2, 'L': 3, 'S': 4, 'H': 5, 'R': 6, 'P': 7, 'A': 8,
                             'W': 9, 'F': 10, 'D': 11, 'C': 12, 'T': 13, 'N': 14, 'V': 15, 'G': 16,
                             'Q': 17, 'K': 18, 'Y': 19, 'E': 20, 'U': 21, 'Z':22, 'B': 23}
species_to_int = {'Other': 0, 'SP': 1, 'MT': 2, 'CH': 3,'TH': 4}

path2 = 'swissprot_annotated_proteins.tab'
def load_data():
    name_sequences = []
    tags_dictionary = {}
    species_dictionary = {}
    with open(path2, 'r') as f:
        for line in f:
            tag = []
            columns = line.strip().split('\t')
            for _ in range(int(columns[2])):
                tag.append(1)
            for _ in range(200 - int(columns[2])):
                tag.append(0)
            tags_dictionary[columns[0]] = tag
            species_dictionary[columns[0]] = species_to_int[columns[1]]
            name_sequences.append(columns[0])
    tags_sequences = [tags_dictionary[name] for name in name_sequences]
    species_sequences = [species_dictionary[name] for name in name_sequences]
    return name_sequences, tags_sequences, species_sequences
def creat_data():
    name, tag, species = load_data()
    name = np.array_split(np.array(name), 5)
    tag = np.array_split(np.array(tag), 5)
    species = np.array_split(np.array(species), 5)
    data1 = TPDataset(name[0], torch.tensor(tag[0]), torch.tensor(species[0]))
    data2 = TPDataset(name[1], torch.tensor(tag[1]), torch.tensor(species[1]))
    data3 = TPDataset(name[2], torch.tensor(tag[2]), torch.tensor(species[2]))
    data4 = TPDataset(name[3], torch.tensor(tag[3]), torch.tensor(species[3]))
    data5 = TPDataset(name[4], torch.tensor(tag[4]), torch.tensor(species[4]))
    return data1, data2, data3, data4, data5

#Pre-reading ISM features
def pre_read():

    names= []
    with open(path2, 'r') as f:
        for line in f:
            columns = line.strip().split('\t')
            names.append(columns[0])
    tensor_dict = {}
    num = 0
    for name in names:
        df = pd.read_feather('ISM_data/' + name + '.feather')
        tensor_dict[name] = torch.tensor(np.array(df.to_numpy()), dtype=torch.float32).to('cuda')
        num += 1
        print(f'\rProgress: {num}/13005', end='', flush=True)
        del df
    print('\n')
    return tensor_dict