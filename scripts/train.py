import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys
import time
from tqdm import tqdm


# Dodajemy folder główny do ścieżki, żeby python widział folder src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.model import get_model

# Parametry
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5  # Na początek 3 dla lepszych wyników mozna więcej
DATA_DIR = os.path.join(project_root, "data", "raw", "chest_xray")
MODEL_SAVE_DIR = os.path.join(project_root, "models")
MODEL_NAME = "pneumonia_resnet.pth"


def train_model():
    #  Wybór urządzenia GPU lub CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    #  Transformacje danych (Augmentacja)
    # Losowe zmiany aby zapobiec overfitingowi
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet wymaga 224x224
        transforms.RandomHorizontalFlip(),  # Odbicie lustrzane
        transforms.RandomRotation(10),  # Lekki obrót
        transforms.ToTensor(),  # Zamiana na macierz liczb
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizacja ImageNet
    ])

    # Dla walidacji
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #Ładowanie danych
    print(f"Szukanie danych w: {DATA_DIR}")

    try:
        train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_transforms)
        val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_transforms)
        # Uzywamy folderu test dlatego ze ma wiecej danych
        test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), val_transforms)
    except FileNotFoundError:
        print("BŁĄD: Nie znaleziono folderów train/test. Uruchom najpierw download_data.py!")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)  # Używamy test jako val

    print(f"Liczba zdjęć treningowych: {len(train_dataset)}")
    print(f"Liczba zdjęć walidacyjnych: {len(test_dataset)}")
    print(f"Klasy: {train_dataset.classes}")

    #  Inicjalizacja modelu
    model = get_model(device)

    # Funkcja kosztu i Optymalizator
    # CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    # Adam algorytm optymalizacji
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Pętla treningowa
    print("\nROZPOCZYNAM TRENING")

    for epoch in range(EPOCHS):
        print(f"\nEpoka {epoch + 1}/{EPOCHS}")
        print("-" * 20)

        # Faza Uczenia
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc="Trening"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zerowanie gradientów
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Walidacja
        model.eval()  # Wyłącza dropout itp.
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():  # Nie liczymy gradientów (oszczędność pamięci)
            for inputs, labels in tqdm(val_loader, desc="Walidacja"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(test_dataset)
        val_acc = val_corrects.double() / len(test_dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    #  Zapis modelu
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel zapisany w: {save_path}")


if __name__ == "__main__":
    train_model()