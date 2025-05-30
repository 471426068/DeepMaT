# ISM

By [Jeffrey Ouyang-Zhang](https://jozhang97.github.io/), [Chengyue Gong](https://sites.google.com/view/chengyue-gong), [Yue Zhao](https://zhaoyue-zephyrus.github.io), [Philipp Krähenbühl](http://www.philkr.net/), [Adam Klivans](https://www.cs.utexas.edu/users/klivans/), [Daniel J. Diaz](http://danny305.github.io)

This repository contains the model presented in the paper [Distilling Structural Representations into Protein Sequence Models](https://www.biorxiv.org/content/10.1101/2024.11.08.622579v1).
The official github can be found at https://github.com/jozhang97/ism.

**TL; DR.** ESM2 with enriched structural representations

## Quickstart

This quickstart assumes that the user is already working with ESM2 and is interested in replacing ESM with ISM. First, download ISM.
```bash
# recommended
huggingface-cli download jozhang97/ism_t33_650M_uc30pdb --local-dir /path/to/save/ism

# alternative
git clone https://huggingface.co/jozhang97/ism_t33_650M_uc30pdb
```

If the user is starting from [fair-esm](https://github.com/facebookresearch/esm), add the following lines of code.
```python
import esm
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
ckpt = torch.load('/path/to/ism_t33_650M_uc30pdb/checkpoint.pth')
model.load_state_dict(ckpt)
```

If the user is starting from [huggingface](https://huggingface.co/facebook/esm2_t33_650M_UR50D), replace the model and tokenizer with the following line of code.
```python
from transformers import AutoTokenizer, AutoModel
config_path = "/path/to/ism_t33_650M_uc30pdb/"
model = AutoModel.from_pretrained(config_path)
tokenizer = AutoTokenizer.from_pretrained(config_path)
```

Please change `/path/to/ism_t33_650M_uc30pdb` to the path where the model is downloaded.

## Citing ISM
If you find ISM useful in your research, please consider citing:

```bibtex
@article{ouyangzhang2024distilling,
  title={Distilling Structural Representations into Protein Sequence Models},
  author={Ouyang-Zhang, Jeffrey and Gong, Chengyue and Zhao, Yue and Kr{\"a}henb{\"u}hl, Philipp and Klivans, Adam and Diaz, Daniel J},
  journal={bioRxiv},
  doi={10.1101/2024.11.08.622579},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
