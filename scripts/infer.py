import torch
from torchvision import transforms
from PIL import Image
import sys
import os


# Dodajemy ścieżkę do src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.model import get_model

# Ścieżka do modelu i klasy
MODEL_PATH = os.path.join(project_root, "models", "pneumonia_resnet.pth")
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']


def predict_image(image_path):
    print(f"\nDIAGNOZA DLA: {os.path.basename(image_path)} ")

    # Sprawdzenie czy model istnieje
    if not os.path.exists(MODEL_PATH):
        print("Błąd: Nie znaleziono modelu. Uruchom najpierw train.py")
        return

    #  Przygotowanie modelu
    device = torch.device("cpu")
    model = get_model(device)

    # Ładujemy wagi
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Błąd ładowania modelu: {e}")
        return

    #  Przygotowanie zdjęcia
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        #  Predykcja
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            score, predicted_idx = torch.max(probabilities, 1)

            label = CLASS_NAMES[predicted_idx.item()]
            confidence = score.item() * 100

        #  Wynik
        print(f"WYNIK: {label}")
        print(f"PEWNOŚĆ: {confidence:.2f}%")

        #
        if label == 'PNEUMONIA':
            print("ZALECENIE: Konsultacja lekarska wymagana.")
        else:
            print("ZALECENIE: Płuca wyglądają na czyste.")

    except Exception as e:
        print(f"Błąd przetwarzania obrazu: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
        predict_image(target_image)
    else:
        print("Użycie: python scripts/infer.py <ścieżka_do_zdjęcia>")
        print("Przykład: python scripts/infer.py data/samples/test_image.jpg")