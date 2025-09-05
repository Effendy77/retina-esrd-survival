
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class FundusSurvivalDataset(Dataset):
    def __init__(self, df, image_root='', img_size=448, time_col='time_to_event', event_col='event_occurred',
                 tabular_cols=None, tabular_norm='standardize', fit_stats=False, stats=None, flip_right_eye=True):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.img_size = img_size
        self.time_col = time_col
        self.event_col = event_col
        self.tabular_cols = tabular_cols or []
        self.tabular_norm = tabular_norm
        self.stats_ = stats
        self.flip_right_eye = flip_right_eye
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        if fit_stats:
            self.stats_ = {}
            for c in self.tabular_cols:
                x = self.df[c].astype(float).values
                m = np.nanmean(x); s = np.nanstd(x) + 1e-8
                self.stats_[c] = (m, s)
    def __len__(self): return len(self.df)
    def _resolve_path(self, p):
        return p if os.path.isabs(p) else os.path.join(self.image_root, p)
    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.flip_right_eye and '21016' in os.path.basename(path):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
    def _load_tabular(self, row):
        if not self.tabular_cols: return None
        xs = []
        for c in self.tabular_cols:
            v = row.get(c, np.nan)
            if pd.isna(v):
                v = self.stats_[c][0] if self.stats_ and c in self.stats_ else 0.0
            if self.tabular_norm == 'standardize' and self.stats_ and c in self.stats_:
                m, s = self.stats_[c]; v = (float(v) - m) / s
            xs.append(float(v))
        return torch.tensor(xs, dtype=torch.float32)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row['image_path'])
        img = self._load_image(img_path)
        img = self.tf(img)
        t = float(row[self.time_col]); e = float(row[self.event_col])
        tab = self._load_tabular(row)
        return {'image': img, 'time': t, 'event': e, 'tab': tab}
