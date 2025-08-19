import torch

# Vérifier si CUDA est disponible
if torch.cuda.is_available():
    print(f"GPU détecté : {torch.cuda.get_device_name(0)}")
    print(f"Nombre de GPUs disponibles : {torch.cuda.device_count()}")
else:
    print("Aucun GPU détecté, exécution sur CPU.")