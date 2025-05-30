# DeepMaT
DeepMaT is a model that simultaneously predicts the classification of targeted peptides and their cleavage sites.


1. Clone project .
```
git clone https://github.com/471426068/DeepMaT.git
cd DeepMaT
```

2. Create conda environment using `environment.yml` file.
```
conda env create -f environment.yml
``` 

3. Train model
```
python train_model.py
```

4. Test model
```
python sample_model.py
```
# Note

ISM model files are required for full operation, just download `checkpoint.pth` and `model.safetensors` into the ism directory at https://huggingface.co/jozhang97/ism_t33_650M_uc30pdb.
