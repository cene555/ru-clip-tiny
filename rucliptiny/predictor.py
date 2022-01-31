import torch
from PIL import Image
from .utils import get_transform


class Predictor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.transform = get_transform()

    def prepare_images_features(self, model, images_path, device='cpu'):
        images_features = []
        for image_path in images_path:
            image = Image.open(image_path)
            image = self.transform(image)
            with torch.no_grad():
                image_features = model.encode_image(image.unsqueeze(0).to(device)).float().cpu()[0]
            images_features.append(image_features)
        images_features = torch.stack(images_features, axis=0)
        return images_features.cpu()

    def prepare_text_features(self, model, texts, max_len=77, device='cpu'):
        texts_features = []
        for text in texts:
            tokens = self.tokenizer.tokenize([text], max_len)
            with torch.no_grad():
                text_features = model.encode_text(tokens[0].to(device), tokens[1].to(device)).float().cpu()[0]
            texts_features.append(text_features)
        texts_features = torch.stack(texts_features, axis=0)
        texts_features /= texts_features.norm(dim=-1, keepdim=True)
        return texts_features

    def __call__(self, model, images_path, classes, get_probs=False, max_len=77, device='cpu'):
        model.eval().to(device)
        image_features = self.prepare_images_features(model, images_path, device)
        texts_features = self.prepare_text_features(model, classes, max_len, device)
        text_probs = (1 * image_features @ texts_features.T).softmax(dim=-1)
        if get_probs:
            return text_probs
        else:
            return text_probs.argmax(-1)