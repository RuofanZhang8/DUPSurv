from pathlib import Path
import os
import json
import pickle
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_pkl(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

class TCGADataset(Dataset):
    """Dataset with tumor presence labels in text"""

    def __init__(self, args, transform, tokenizer, max_length, text_encoder, fold=0, **kwargs):
        split = kwargs["split"]
        self.crop_size = args.resolution
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_encoder = text_encoder.to('cpu')
        
        self.ft_root = kwargs["ft_root"]
        self.split_data_root = kwargs.get('split_data_root', '/path/to/data_split')

        # Load WSI dataset
        data_file = os.path.join(self.split_data_root, f'TCGA_BRCA_overall_survival_k={fold}', f'{split}.npz')
        loaded_data = np.load(data_file, allow_pickle=True)['data']
        
        self.pid = loaded_data[:, 0]
        self.svs_name = loaded_data[:, 1]
        self.img_path = loaded_data[:, 2]
        self.tissue_type = loaded_data[:, 3]

        # Load report and tissue dictionaries
        self.report_dict = load_json('/path/to/pid_report.json')
        self.tissue_dict = load_json('/path/to/tissue_class.json')
        
        self.tissue_dict_embedding = {k: self.get_embedding(v) for k, v in self.tissue_dict.items()}

        # Load gene set embeddings dictionary
        csv_path = os.path.join(self.split_data_root, f'TCGA_BRCA_overall_survival_k={fold}', f'{split}.csv')
        self.gs_emb_dict = self.get_gs_emb_dict(csv_path)

        # Load precomputed embeddings
        emb_file = os.path.join(self.split_data_root, f'TCGA_BRCA_overall_survival_k={fold}', 'embeddings',
                                'precomputed_embeddings.pkl')
        gs_embedding = load_pkl(emb_file)
        self.gs_embedding_mean = gs_embedding[split]['mean']
        self.gs_embedding_cov = gs_embedding[split]['cov']

    def __len__(self):
        return len(self.img_path)
    
    def get_gs_emb_dict(self, csv_path):
        import pandas as pd
        sample_col = 'case_id'
        df = pd.read_csv(csv_path)
        idx2sample_df = pd.DataFrame({'sample_id': df[sample_col].astype(str).unique()})
        return {row["sample_id"]: index for index, row in idx2sample_df.iterrows()}

    def tokenize_captions(self, captions):
        inputs = self.tokenizer(
            captions, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    @torch.no_grad()
    def get_embedding(self, captions):
        text_input_ids = self.tokenizer(captions, return_tensors="pt", padding="max_length", truncation=True).input_ids
        prompt_embeds = self.text_encoder(text_input_ids, attention_mask=None)
        return prompt_embeds[0]

    @staticmethod
    def get_random_crop(img, size):
        x = np.random.randint(0, img.shape[1] - size)
        y = np.random.randint(0, img.shape[0] - size)
        return img[y : y + size, x : x + size]

    def __getitem__(self, idx):
        example = {}
        image = Image.open(self.img_path[idx]).convert("RGB")
        example["pixel_values"] = self.transform(image)
        example['report_ids'] = self.tokenize_captions(self.report_dict[self.pid[idx]])
        example["tissue_embedding"] = self.tissue_dict_embedding[self.tissue_type[idx]]
        
        slide_embedding = torch.load(os.path.join(self.ft_root, f'{self.svs_name[idx]}.pt'), weights_only=False)
        example['slide_embedding'] = slide_embedding
        
        slide_idx = self.gs_emb_dict[self.pid[idx]]
        example['gs_embedding'] = self.gs_embedding_mean[slide_idx]

        return example
