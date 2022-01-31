from transformers import DistilBertTokenizer
import torch


class Tokenizer:
    def __init__(self):
        tokenizer_load = "DeepPavlov/distilrubert-tiny-cased-conversational-v1"
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_load)
    
    def tokenize(self, texts, max_len=77):
        tokenized = self.tokenizer.batch_encode_plus(texts,
                                                    truncation=True,
                                                    add_special_tokens = True,
                                                    max_length = max_len,
                                                    padding='max_length',
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt')
        return torch.stack([tokenized["input_ids"], tokenized["attention_mask"]])