# flame_model_loader.py
import pickle

def load_flame_model(path="models/FLAME_2020_model.pkl"):
    with open(path, "rb") as f:
        flame_model = pickle.load(f, encoding="latin1")
    print("FLAME model keys:", flame_model.keys())
    return flame_model
