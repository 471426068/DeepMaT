import torch
from DeepMaT import tool
from DeepMaT.tool import ISM_Feature
from DeepMaT.model.composite_model import ProteinModel
from matplotlib import pyplot as plt

tool.set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sequence = ['MDVASLISPSESDTVPTFRSRSIQNSSASHYKRLSEQSTGSYFSAVPTHTTSYSRTPQPPLSPPAEDQSKCSLPSISILLENADGAAAHAAKRQRNSLSTHRDSDPRPPYDSITPHAMPPTPPLRPGSGFHSNGHSPSTSSVSAASSSALMKNTESYPQAPIGLPSPTDRSSISSQGSVQHAASAPYASPAPSVSSFSSPIEPSTPSTAAYYQRNPAPNTFQNPSPFPQTSTASLPSPGHQQMISPVTPAWQHHHYFPPSSSTSYQQNHDRYICRTCHKAFSRPSSLRIHSHSHTGEKPFRCTHAGCGKAFSVRSNMKRHERGCHTGRPVATAMVQ',
            'MSAQTASGPTEDQVEILEYNFNKVNKHPDPTTLCLIAAEAGLTEEQTQKWFKQRLAEWRRSEGLPSECRSVTD']

name = [['seq'+str(i)]  for i in range(len(sequence))]

seq_Feature = torch.tensor(ISM_Feature(sequence), dtype=torch.float32, device=device)

model = ProteinModel().to(device)
model_params = torch.load('../model/model0.pth', map_location=device)
model.load_state_dict(model_params)

model.eval()
cls_pre, crf_pre = model(seq_Feature)
cls_pre = torch.softmax(cls_pre, dim=1).detach().cpu().numpy()
TP_class = ['Other', 'SP', 'MT', 'CH', 'TH']
for i in range(len(cls_pre)):
    plt.title(str(name[i]))
    plt.bar(TP_class, cls_pre[i])
    plt.show()
    print(name[i], ':', crf_pre.squeeze(0).detach().cpu().numpy()[i])


