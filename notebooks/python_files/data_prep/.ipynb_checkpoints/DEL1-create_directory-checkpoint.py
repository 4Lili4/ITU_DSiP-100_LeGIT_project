import os

# shutil.rmtree("./artifacts",ignore_errors=True)

def cr_art_dir():
    os.makedirs("artifacts",exist_ok=True)
    print("Created artifacts directory")