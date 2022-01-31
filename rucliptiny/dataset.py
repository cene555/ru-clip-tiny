import cv2
from torch.utils.data import Dataset
import transformers
from .utils import get_transform
from .tokenizer import Tokenizer
import pandas as pd
import os
from PIL import Image


class RuCLIPTinyDataset(Dataset):
    def __init__(self, dir, df_path, max_text_len=77):
        self.df = pd.read_csv(df_path)
        self.dir = dir
        self.max_text_len = max_text_len
        self.tokenizer = Tokenizer()
        self.transform = get_transform()

    def __getitem__(self, idx):
        # достаем имя изображения и ее лейбл
        image_name = self.df['image_name'].iloc[idx]
        text = self.df['text'].iloc[idx]
        tokens = self.tokenizer.tokenize([text], max_len=self.max_text_len)
        input_ids, attention_mask = tokens[0][0], tokens[1][0]
        image = cv2.imread(os.path.join(self.dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image)
        return image, input_ids, attention_mask

    def __len__(self):
        return len(self.df)
