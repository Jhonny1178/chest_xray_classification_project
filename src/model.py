import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def get_model(device='cpu'):
    #Tworzymy model ResNet18 dostosowany do klasyfikacji binarnej (2 klasy).
    # Używa wag pretrenowanych na ImageNet (Transfer Learning).

    print("Inicjalizacja modelu ResNet18")

    # Pobieramy model z najlepszymi dostępnymi wagami
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Zamrażanie wag (opcjonalne - tu zostawiamy odmrożone dla lepszego dopasowania)
    # To pozwala sieci "nauczyć się" specyfiki zdjęć RTG

    # Podmieniamy ostatnią warstwę (Fully Connected)
    # Oryginalny ResNet ma 1000 wyjść (klas ImageNet). My potrzebujemy 2:
    # 0: NORMAL
    # 1: PNEUMONIA
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(device)
    return model