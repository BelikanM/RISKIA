import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

image = Image.new('RGB', (224,224), color='red')
texts = ['test risk']

inputs = processor(text=texts, images=image, return_tensors='pt', padding=True).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    print('CLIP test successful, shape:', outputs.logits_per_image.shape)