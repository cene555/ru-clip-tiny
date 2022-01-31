from timm import create_model
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import DistilBertModel, DistilBertConfig


class RuCLIPtiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = create_model('convnext_tiny',
                                   pretrained=False,
                                   num_classes=0,
                                   in_chans=3)  # out 768
        text_config = DistilBertConfig(**{"vocab_size": 30522,
                                          "max_position_embeddings": 512,
                                          "n_layers": 3,
                                          "n_heads": 12,
                                          "dim": 264,
                                          "hidden_dim": 792,
                                          "model_type": "distilbert"})
        self.transformer = DistilBertModel(text_config)
        self.final_ln = torch.nn.Linear(264, 768)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        return self.visual.stem[0].weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.final_ln(x)
        return x

    def forward(self, image, input_ids, attention_mask):
        image_features = self.encode_image(image)
        text_features = self.encode_text(input_ids, attention_mask)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
