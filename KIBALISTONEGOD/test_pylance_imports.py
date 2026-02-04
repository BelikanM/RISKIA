# Test Pylance - Fichier de vérification des imports
import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import open3d as o3d
import transformers
import sklearn

# Test rapide
print("✅ Tous les imports fonctionnent!")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")