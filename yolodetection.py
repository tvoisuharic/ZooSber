import subprocess
import sys
import os
import requests
from ultralytics import YOLO

def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def get_animal_from_image(img_path: str) -> str:
    model = YOLO('yolo11x-cls.pt')
    results = model(img_path)

    if not results[0].probs:
        return "Животное не определено"

    top_class = results[0].probs.top1
    top_prob = results[0].probs.top1conf
    animal_name = model.names[top_class]

    return f'С вероятностью {top_prob:.2f} на картинке: {animal_name}'

def main():
    install_packages()
    print(get_animal_from_image("корова.jpg"))
if __name__ == "__main__":
    main()
