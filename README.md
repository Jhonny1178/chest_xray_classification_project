Autor: Jan Nowojski 

1. Opis
Projekt wykorzystuje sieć neuronową ResNet18 (Transfer Learning) do klasyfikacji zdjęć RTG klatki piersiowej na dwie kategorie: NORMAL-zdrowy i PNEUMONIA-chory

2. Instalacja środowiska
Aby uruchomić projekt zainstaluj wymagane biblioteki:
pip install -r requirements.txt

3. Przygotowanie danych
Projekt posiada automat do pobierania danych z Kaggle.
python scripts/download_data.py

4. Trening modelu
Uruchomienie treningu. Skrypt zapisze model w folderze:models/
python scripts/train.py

5. Inferencja
Aby sprawdzić pojedyncze zdjęcie (przykłady znajdują się w data/samples):
python scripts/infer.py data/samples/NORMAL.jpeg 
python scripts/infer.py data/samples/PNEUMONIA.jpeg 
 