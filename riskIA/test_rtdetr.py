import torch
from transformers import pipeline
from PIL import Image

print('Torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

# Create a test image
image = Image.new('RGB', (640, 480), color='gray')

# Load RT-DETR
detector = pipeline("object-detection", model="PekingU/rtdetr_r50vd_coco_o365", device=0 if torch.cuda.is_available() else -1)

# Detect objects
detections = detector(image, threshold=0.5)
print('Detections:', detections)

# Extract unique labels
detected_objects = list(set([det['label'] for det in detections]))
print('Detected objects:', detected_objects)