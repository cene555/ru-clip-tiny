# RuCLIPtiny
Zero-shot image classification model for Russian language

---

RuCLIP tiny (Russian Contrastive Language–Image Pretraining) is a neural network trained to work with different pairs (images, texts). Our model is based on [ConvNeXt-tiny](https://github.com/facebookresearch/ConvNeXt) and [DistilRuBert-tiny](https://huggingface.co/DeepPavlov/distilrubert-tiny-cased-conversational-v1), and is supported by extensive research zero-shot transfer, computer vision, natural language processing, and multimodal learning.

## Result evaluation

Our model achieved 46.62% top1 and 73.18% top5 zero-shot accuracy on CIFAR100

## Examples

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-l2JtynS3ZwlE8g5wNYNdTYUVQRLWl9m?usp=sharing)
Evaluate & Simple usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lojdsARMzsURUkKJQLlEQAPyvM99U12n?usp=sharing)
Finetuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Yl1oVem3Pw0o1ZlukR2Fg9dGyqRFeu1z?usp=sharing)
ONNX conversion and speed testing

### [Model weights](https://drive.google.com/uc?id=1-3g3J90pZmHo9jbBzsEmr7ei5zm3VXOL)

## Usage
Install rucliptiny module and requirements first. Use this trick
```python3
!gdown -O ru-clip-tiny.pkl https://drive.google.com/uc?id=1-3g3J90pZmHo9jbBzsEmr7ei5zm3VXOL
!pip install git+https://github.com/cene555/ru-clip-tiny.git
```
## Example in 3 steps
Download CLIP image from repo
```python3
!wget -c -O CLIP.png https://github.com/openai/CLIP/blob/main/CLIP.png?raw=true
```
1. Import libraries
```python3
from rucliptiny.predictor import Predictor
from rucliptiny import RuCLIPtiny
import torch

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
2. Load model
```python3
model = RuCLIPtiny()
model.load_state_dict(torch.load('ru-clip-tiny.pkl'))
model = model.to(device).eval()
```
3. Use predictor to get probabilities
```python3
predictor = Predictor()

classes = ['диаграмма', 'собака', 'кошка']
text_probs = predictor(model=model, images_path=["CLIP.png"],
                       classes=classes, get_probs=True,
                       max_len=77, device=device)
```

## Cosine similarity Visualization Example

![image](https://github.com/cene555/ru-clip-tiny/blob/main/pictures/similarity.jpg)

## Speed Testing

NVIDIA Tesla K80 (Google Colab session)

| TORCH      |   batch |   encode_image |   encode_text |   total |
|:-----------|--------:|---------------:|--------------:|--------:|
| RuCLIPtiny |       2 |          0.011 |         0.004 |   0.015 |
| RuCLIPtiny |       8 |          0.011 |         0.004 |   0.015 |
| RuCLIPtiny |      16 |          0.012 |         0.005 |   0.017 |
| RuCLIPtiny |      32 |          0.014 |         0.005 |   0.019 |
| RuCLIPtiny |      64 |          0.013 |         0.006 |   0.019 |

---

We would like to express my gratitude to Sber AI for the grants provided,
for which research was carried out, as part of the Artificial Intelligence International Junior Contest (AIIJC)
