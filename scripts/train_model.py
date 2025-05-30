import gc
import time
from DeepMaT import tool
import torch
from torch.utils.data import DataLoader

from DeepMaT.data.load_data import creat_data, pre_read
from DeepMaT.model.composite_model import ProteinModel
from DeepMaT.tool import species_to_one_hot


def train(train_model, train_optimizer,train_loader):

    for n, (batch_x, batch_y, batch_species), in enumerate(train_loader):
        seq_sequences = torch.tensor([]).to('cuda')
        for name_id in batch_x:
            if not seq_sequences.numel():
                seq_sequences = ism_data[name_id].unsqueeze(0)
            else:
                seq_sequences = torch.cat([seq_sequences, ism_data[name_id].unsqueeze(0)])
            torch.cuda.empty_cache()

        batch_species = torch.tensor(species_to_one_hot(batch_species))
        batch_y, batch_species = batch_y.to(device), batch_species.to(device)

        train_optimizer.zero_grad()
        cls_out, crf_loss = train_model(seq_sequences, batch_y)
        cls_loss = tool.cross_entropy_loss_with_soft_target(cls_out, batch_species)
        loss = crf_loss + cls_loss
        loss.backward()
        train_optimizer.step()
    return train_model

tool.set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data1, data2, data3, data4, data5 = creat_data()
prepare_data = time.time()
ism_data = pre_read()
print('\nprepare_data_time:',time.time()-prepare_data)
gc.collect()
number = 50
for i in range(5):
    print('\rtrain', i + 1, end=' ', flush=True)
    if i != 0:
        t = data1
        data1 = data2
        data2 = data3
        data3 = data4
        data4 = data5
        data5 = t
    model = ProteinModel().to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
    data1_loader = DataLoader(data1, batch_size=128, shuffle=True,num_workers=8)
    data2_loader = DataLoader(data2, batch_size=128, shuffle=True,num_workers=8)
    data3_loader = DataLoader(data3, batch_size=128, shuffle=True,num_workers=8)
    data4_loader = DataLoader(data4, batch_size=128, shuffle=True,num_workers=8)

    start = time.time()
    for j in range(number):
        model.train()
        model = train(model, optimizer, data1_loader)
        model = train(model, optimizer, data2_loader)
        model = train(model, optimizer, data3_loader)
        model = train(model, optimizer, data4_loader)

    torch.save(model.state_dict(), '../model' + str(i) + '.pth')
    torch.save(optimizer.state_dict(), '../optimizer' + str(i) + '.pth')
    end = time.time()
    print('\ntrain_time:',end-start)