import kagglehub
import os
import shutil

# Link do danych z kaggla
dataset = "paultimothymooney/chest-xray-pneumonia"
script_location = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_location)
target = os.path.join(project_root, "data", "raw", "chest_xray")





# Sprawdzamy, czy dane już istnieją
if os.path.exists(target):
    print(f"Dane już istnieją w folderze: {target}")
else:
    try:
        # Pobieranie
        print("Rozpoczynanie pobierania danych")
        try:
            print(f"Pobieranie zbioru: {dataset}")
            path = kagglehub.dataset_download(dataset)
            print(f"Pobrano dane do folderu tymczasowego: {path}")
        except Exception as e:
            print(f"Błąd pobierania: {e}")


        # Przenoszenie plików do struktury projektu
        print(f"Kopiowanie danych do: {target}")


        try:
            # sprawdzamy, czy chest_xray nie jest w podfolderze
            source_path = os.path.join(path, "chest_xray")
            if not os.path.exists(source_path):
                source_path = path

            shutil.copytree(source_path, target,dirs_exist_ok=True)
            print("Sukces: Dane zostały skopiowane")
        except Exception as e:
            print(f"Błąd podczas kopiowania plików: {e}")

    except Exception as e:
        print(f"Błąd podczas pobierania: {e}")



