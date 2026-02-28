import os

# Base project name
BASE_DIR = "sensex_ml_trading"

# Folder structure
folders = [
    "data/raw",
    "data/processed",
    "src",
    "notebooks",
    "models"
]

# Files to create
files = [
    "src/__init__.py",
    "src/config.py",
    "src/data_loader.py",
    "src/data_preprocessing.py",
    "src/features.py",
    "src/labels.py",
    "src/model.py",
    "src/evaluation.py",
    "src/backtest.py",
    "src/live_predict.py",
    "src/visualization.py",
    "notebooks/exploration.ipynb",
    "models/random_forest.pkl",
    "requirements.txt",
    "README.md",
    "main.py"
]

def create_structure():
    # Create base directory
    os.makedirs(BASE_DIR, exist_ok=True)

    # Create folders
    for folder in folders:
        os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

    # Create files
    for file in files:
        file_path = os.path.join(BASE_DIR, file)
        with open(file_path, "w") as f:
            if file.endswith(".py"):
                f.write("# TODO: Implement this module\n")
            elif file.endswith(".md"):
                f.write("# Sensex ML Trading Project\n")
            else:
                pass  # Empty file

    print("✅ Project structure created successfully!")

if __name__ == "__main__":
    create_structure()