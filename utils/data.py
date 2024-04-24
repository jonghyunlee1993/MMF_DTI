import pickle
import numpy as np
import pandas as pd
from rdkit import Chem

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def load_dataset(mode="DAVIS", unseen_cond="No", missing_cond=0):
    print(mode, unseen_cond, missing_cond)
    
    if mode in ["DAVIS", "BindingDB", "BIOSNAP"] and (unseen_cond == "No") and (missing_cond == 0):
        train_df = pd.read_csv(f"./data/{mode}_train.csv")
        valid_df = pd.read_csv(f"./data/{mode}_valid.csv")
        test_df = pd.read_csv(f"./data/{mode}_test.csv")
    elif mode == "merged":
        train_df = pd.read_csv("./data/train_dataset.csv")
        valid_df = pd.read_csv("./data/valid_dataset.csv")
        test_df = pd.read_csv("./data/test_dataset.csv")
    elif mode == "BIOSNAP" and (unseen_cond in ["drug", "target"]):
        train_df = pd.read_csv(f"./data/BIOSNAP_train_unseen_{unseen_cond}.csv")
        valid_df = pd.read_csv(f"./data/BIOSNAP_valid_unseen_{unseen_cond}.csv")
        test_df = pd.read_csv(f"./data/BIOSNAP_test_unseen_{unseen_cond}.csv")
    elif mode == "BIOSNAP" and (missing_cond in [70, 80, 90, 95]):
        train_df = pd.read_csv(f"./data/BIOSNAP_train_missing_{missing_cond}.csv")
        valid_df = pd.read_csv(f"./data/BIOSNAP_valid_missing_{missing_cond}.csv")
        test_df = pd.read_csv(f"./data/BIOSNAP_test_missing_{missing_cond}.csv")
    
    return train_df, valid_df, test_df
        
        
def load_cached_prot_features(max_length=545):
    with open(f"prot_feat/{max_length}_cls.pkl", "rb") as f:
        prot_feat_teacher = pickle.load(f)
        
    return prot_feat_teacher


class DTIDataset(Dataset):
    def __init__(self, 
                 data, 
                 prot_feat_teacher=None, 
                 mol_tokenizer=None, 
                 mol_max_length=512,
                 prot_tokenizer=None, 
                 prot_max_length=545, 
                 use_text_feat=False,
                 text_tokenizer=None, 
                 text_max_length=512,
                 use_property=False,
                 d_mode="merged", 
                 use_enumeration=False):
        
        self.data = data
        self.prot_feat_teacher = prot_feat_teacher
        self.mol_tokenizer = mol_tokenizer
        self.mol_max_lenght = mol_max_length
        self.prot_tokenizer = prot_tokenizer
        self.prot_max_length = prot_max_length
        self.use_text_feat = use_text_feat
        self.text_tokenizer = text_tokenizer
        self.text_max_length = text_max_length
        self.use_property = use_property
        self.d_mode = d_mode
        self.use_enumeration = use_enumeration
        
    def get_mol_feat(self, smiles):
        return self.mol_tokenizer(smiles, max_length=self.mol_max_lenght, truncation=True)
    
    def get_prot_feat_student(self, fasta):
        return self.prot_tokenizer(" ".join(fasta), max_length=self.prot_max_length + 2, truncation=True)
    
    def get_prot_feat_teacher(self, fasta):
        return self.prot_feat_teacher[fasta[:20]]
    
    def get_fuctional_text(self, text):
        if pd.isna(text):
            text = "Not applicable."
        return self.text_tokenizer(text, max_length=self.text_max_length, truncation=True)
    
    def __len__(self):    
        return len(self.data)
    
    def __getitem__(self, index):
        smiles = self.data.loc[index, "SMILES"]
        mol_feat = self.get_mol_feat(smiles)
        fasta = self.data.loc[index, "Target Sequence"]
        prot_feat_student = self.get_prot_feat_student(fasta)
        prot_feat_teacher = self.get_prot_feat_teacher(fasta)
        
        text_feat = None
        if self.use_text_feat:
            text_feat = self.get_fuctional_text(self.data.loc[index, "sFunction"])
        
        property = None
        if self.use_property:
            property = self.data.iloc[index, 4:].values
        
        y = self.data.loc[index, "Label"]
        
        if self.d_mode == "merged":
            source = self.data.loc[index, "Source"]
            if source == "DAVIS":
                source = 1
            elif source == "BindingDB":
                source = 2
            elif source == "BIOSNAP":
                source = 3
        elif self.d_mode == "DAVIS":
            source = 1
        elif self.d_mode == "BindingDB":
            source = 2
        elif self.d_mode == "BIOSNAP":
            source = 3
                
        return mol_feat, prot_feat_student, prot_feat_teacher, text_feat, property, y, source


class CollateBatch(object):
    def __init__(self, mol_tokenizer, prot_tokenizer, text_tokenizer=None, use_property=False):
        self.mol_tokenizer = mol_tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.text_tokenizer = text_tokenizer
        
        self.use_text_feat = True if self.text_tokenizer is not None else False
        self.use_property = use_property
        
    def __call__(self, batch):
        mol_features, prot_feat_student, prot_feat_teacher, text_feature, property, y, source = [], [], [], [], [], [], []
    
        for (mol_seq, prot_seq_student, prot_seq_teacher, text_feature_, property_, y_, source_) in batch:
            mol_features.append(mol_seq)
            prot_feat_student.append(prot_seq_student)
            prot_feat_teacher.append(prot_seq_teacher.numpy().tolist())
            
            if self.use_text_feat:
                text_feature.append(text_feature_)
            
            if self.use_property:
                property.append(property_.astype(float).tolist())
            
            y.append(y_.astype(int).tolist())
            source.append(source_)
            
        mol_features = self.mol_tokenizer.pad(mol_features, return_tensors="pt")
        prot_feat_student = self.prot_tokenizer.pad(prot_feat_student, return_tensors="pt")
        prot_feat_teacher = torch.tensor(prot_feat_teacher).float()
        
        if self.use_text_feat:
            text_feature = self.text_tokenizer.pad(text_feature, return_tensors="pt")
        
        if self.use_property:
            property = torch.tensor(property).float()
            property = torch.nan_to_num(property)
        
        y = torch.tensor(y).float()
        source = torch.tensor(source)
        
        return mol_features, prot_feat_student, prot_feat_teacher, text_feature, property, y, source
    

def define_balanced_sampler(train_df, target_col_name="Label"):
    counts = np.bincount(train_df[target_col_name])
    labels_weights = 1. / counts
    weights = labels_weights[train_df[target_col_name]]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    return sampler


def get_dataloaders(train_df, 
                    valid_df, 
                    test_df, 
                    prot_feat_teacher=None, 
                    mol_tokenizer=None, 
                    mol_max_length=512,
                    prot_tokenizer=None, 
                    prot_max_length=545, 
                    use_text_feat=False,
                    text_tokenizer=None,
                    text_max_length=512,
                    use_property=False,
                    d_mode="merged", 
                    use_enumeration=False, 
                    use_sampler=False,
                    batch_size=128,
                    num_workers=-1):
    
    train_dataset = DTIDataset(train_df, prot_feat_teacher, 
                               mol_tokenizer, mol_max_length, 
                               prot_tokenizer, prot_max_length, 
                               use_text_feat, text_tokenizer, text_max_length, 
                               use_property, d_mode, use_enumeration)
    valid_dataset = DTIDataset(valid_df, prot_feat_teacher, 
                               mol_tokenizer, mol_max_length, 
                               prot_tokenizer, prot_max_length, 
                               use_text_feat, text_tokenizer, text_max_length, 
                               use_property, d_mode)
    test_dataset = DTIDataset(test_df, prot_feat_teacher, 
                               mol_tokenizer, mol_max_length, 
                               prot_tokenizer, prot_max_length, 
                               use_text_feat, text_tokenizer, text_max_length, 
                               use_property, d_mode)
    
    if use_sampler:
        sampler = define_balanced_sampler(train_df)
    
    collator = CollateBatch(mol_tokenizer, prot_tokenizer, text_tokenizer, use_property)
    
    if not use_sampler:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                      num_workers=num_workers, pin_memory=True,
                                      shuffle=True, collate_fn=collator)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                      num_workers=num_workers, pin_memory=True,
                                      sampler=sampler, collate_fn=collator)
    
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, 
                                  num_workers=num_workers, pin_memory=True,
                                  shuffle=False, collate_fn=collator)
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                 num_workers=num_workers, pin_memory=True,
                                 shuffle=False, collate_fn=collator)
    
    return train_dataloader, valid_dataloader, test_dataloader