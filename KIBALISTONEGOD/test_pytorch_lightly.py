#!/usr/bin/env python3
"""
Test rapide PyTorch + Lightly
"""
import torch
import lightly

print('✅ PyTorch:', torch.__version__)
print('✅ CUDA disponible:', torch.cuda.is_available())
print('✅ Lightly:', lightly.__version__)

# Test rapide GPU si disponible
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
    x = torch.randn(100, 100).cuda()
    print('✅ Calcul GPU réussi')
else:
    print('ℹ️  Pas de GPU CUDA détecté')

print()
print('🎉 TOUTES LES DÉPENDANCES SONT OPÉRATIONNELLES!')